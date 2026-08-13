"""Orchestrate Credit purchases, signed Stripe events, and refunds."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from dashboard.backend.domain.credits.config import load_billing_config
from dashboard.backend.domain.credits.models import (
    AdminRefundRequest,
    BalanceResult,
    CheckoutRequest,
    CheckoutResult,
    RefundCreationResult,
    WebhookResult,
    credits_micro_for_cents,
    format_credits,
)
from dashboard.backend.domain.credits.repository import (
    RefundNotAllowedError,
    credits_store,
)
from dashboard.backend.domain.credits.stripe_gateway import (
    StripeGatewayError,
    StripeTestGateway,
    StripeWebhookEvent,
)


class CreditsServiceError(RuntimeError):
    """A sanitized, expected billing-domain failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class PaymentOrderNotFoundError(CreditsServiceError):
    def __init__(self):
        super().__init__("payment_order_not_found", "Payment order was not found")


def _operation_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256(
        "\x1f".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _required_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _stripe_object_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return _required_text(value.get("id"))
    return _required_text(value)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class CreditsService:
    def __init__(self, *, store=None, gateway=None):
        self.store = store or credits_store
        self.gateway = gateway or StripeTestGateway(load_billing_config())

    def get_balance(self, user_id: int) -> BalanceResult:
        self.store.ensure_account(user_id)
        balance = self.store.get_balance_micro(user_id)
        return BalanceResult(
            balance_micro=balance,
            display_credits=format_credits(balance),
        )

    def create_checkout(self, user_id: int, request: CheckoutRequest) -> CheckoutResult:
        amount = request.amount_usd_cents
        credits_micro = credits_micro_for_cents(amount)
        order_id = _operation_id("ord", user_id, request.client_request_id)
        order = self.store.create_or_get_order(
            order_id=order_id,
            user_id=user_id,
            client_request_id=str(request.client_request_id),
            amount_usd_cents=amount,
            credits_micro=credits_micro,
        )

        session = self.gateway.create_checkout_session(
            order_id=order["id"],
            user_reference=str(user_id),
            amount_usd_cents=order["amount_usd_cents"],
            credits_micro=order["credits_micro"],
            idempotency_key=f"checkout:{order['id']}",
        )
        updated = self.store.attach_checkout_session(
            order["id"], checkout_session_id=session.session_id
        )
        return CheckoutResult(
            order_id=updated["id"],
            checkout_session_id=session.session_id,
            checkout_url=session.checkout_url,
            amount_usd_cents=updated["amount_usd_cents"],
            credits_micro=updated["credits_micro"],
            order_status=updated["status"],
        )

    def create_admin_refund(
        self, admin_user_id: int, request: AdminRefundRequest
    ) -> RefundCreationResult:
        order = self.store.get_order_for_admin(request.payment_order_id)
        if not order:
            raise PaymentOrderNotFoundError()
        payment_intent_id = _required_text(order.get("stripe_payment_intent_id"))
        if not payment_intent_id:
            raise CreditsServiceError(
                "purchase_not_refundable", "Purchase has no settled payment"
            )

        refund_id = _operation_id(
            "rfnd",
            admin_user_id,
            request.payment_order_id,
            request.client_request_id,
        )
        credits_micro = credits_micro_for_cents(request.amount_usd_cents)
        reservation = self.store.reserve_refund(
            refund_id=refund_id,
            payment_order_id=order["id"],
            user_id=order["user_id"],
            requested_by_user_id=admin_user_id,
            amount_usd_cents=request.amount_usd_cents,
            credits_micro=credits_micro,
        )
        result = self.gateway.create_refund(
            refund_id=reservation["id"],
            payment_intent_id=payment_intent_id,
            amount_usd_cents=reservation["amount_usd_cents"],
            idempotency_key=f"refund:{reservation['id']}",
        )
        if (
            result.payment_intent_id != payment_intent_id
            or result.amount_usd_cents != reservation["amount_usd_cents"]
        ):
            raise StripeGatewayError("Stripe returned a mismatched Refund")
        attached = self.store.attach_stripe_refund(
            reservation["id"], stripe_refund_id=result.refund_id
        )
        return RefundCreationResult(
            refund_id=attached["id"],
            stripe_refund_id=attached["stripe_refund_id"],
            payment_order_id=attached["payment_order_id"],
            amount_usd_cents=attached["amount_usd_cents"],
            credits_micro=attached["credits_micro"],
            refund_status=attached["status"],
        )

    def handle_webhook(self, payload: bytes, signature_header: str) -> WebhookResult:
        event = self.gateway.verify_webhook(payload, signature_header)
        if event.livemode:
            return self._record_event(
                event,
                outcome="rejected",
                reason="Stripe Live Mode events are not accepted",
            )
        if event.event_type == "checkout.session.completed":
            return self._handle_checkout_completed(event)
        if event.event_type in {"refund.created", "refund.updated", "refund.failed"}:
            return self._handle_refund_event(event)
        return self._record_event(
            event,
            outcome="ignored",
            reason="Unsupported Stripe event type",
        )

    def _record_event(
        self,
        event: StripeWebhookEvent,
        *,
        outcome: str,
        reason: str,
        account_restricted: bool = False,
    ) -> WebhookResult:
        stored = self.store.record_webhook_event(
            event_id=event.event_id,
            event_type=event.event_type,
            livemode=event.livemode,
            object_id=event.object_id,
            payload_sha256=event.payload_sha256,
            outcome=outcome,
            reason=reason,
        )
        return WebhookResult(
            outcome=stored["outcome"],
            event_type=event.event_type,
            reason=stored.get("reason"),
            account_restricted=account_restricted,
        )

    def _handle_checkout_completed(self, event: StripeWebhookEvent) -> WebhookResult:
        obj = event.data_object
        metadata = obj.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        order_id = _required_text(metadata.get("atl_order_id"))
        client_reference_id = _required_text(obj.get("client_reference_id"))
        payment_status = _required_text(obj.get("payment_status"))

        if event.livemode:
            return self._record_event(
                event, outcome="rejected", reason="Live Mode payment is not accepted"
            )
        if payment_status != "paid":
            return self._record_event(
                event, outcome="ignored", reason="Checkout is not paid"
            )
        if not order_id or client_reference_id != order_id:
            return self._record_event(
                event, outcome="rejected", reason="Checkout order metadata is invalid"
            )

        order = self.store.get_order_for_admin(order_id)
        if not order:
            return self._record_event(
                event, outcome="rejected", reason="Payment order was not found"
            )
        expected_user = str(order["user_id"])
        if _required_text(metadata.get("atl_user_reference")) != expected_user:
            return self._record_event(
                event, outcome="rejected", reason="Checkout user metadata is invalid"
            )
        if _required_text(metadata.get("atl_credits_micro")) != str(
            order["credits_micro"]
        ):
            return self._record_event(
                event, outcome="rejected", reason="Checkout Credit metadata is invalid"
            )

        amount = _integer(obj.get("amount_total"))
        currency = _required_text(obj.get("currency"))
        payment_intent = _stripe_object_id(obj.get("payment_intent"))
        if amount is None or not currency or not payment_intent:
            return self._record_event(
                event, outcome="rejected", reason="Checkout payment data is incomplete"
            )
        result = self.store.settle_paid_checkout(
            event_id=event.event_id,
            event_type=event.event_type,
            livemode=event.livemode,
            object_id=event.object_id,
            payload_sha256=event.payload_sha256,
            order_id=order_id,
            checkout_session_id=event.object_id,
            payment_intent_id=payment_intent,
            currency=currency,
            amount_usd_cents=amount,
        )
        return WebhookResult(
            outcome=result["outcome"],
            event_type=event.event_type,
            reason=result.get("reason"),
            balance_micro=result.get("balance_micro"),
        )

    def _handle_refund_event(self, event: StripeWebhookEvent) -> WebhookResult:
        obj = event.data_object
        metadata = obj.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        local_refund_id = _required_text(metadata.get("atl_refund_id"))
        stripe_refund_id = event.object_id
        payment_intent_id = _stripe_object_id(obj.get("payment_intent"))
        amount = _integer(obj.get("amount"))
        currency = _required_text(obj.get("currency"))
        status = (_required_text(obj.get("status")) or "").lower()

        if event.livemode:
            return self._record_event(
                event, outcome="rejected", reason="Live Mode refund is not accepted"
            )
        if not payment_intent_id or amount is None or not currency:
            return self._record_event(
                event, outcome="rejected", reason="Refund payment data is incomplete"
            )

        refund = (
            self.store.get_refund_by_id(local_refund_id) if local_refund_id else None
        )
        if not refund:
            refund = self.store.get_refund_by_stripe_id(stripe_refund_id)

        if not refund and status == "succeeded":
            refund = self._reconcile_external_refund(
                event,
                payment_intent_id=payment_intent_id,
                amount_usd_cents=amount,
                currency=currency,
            )
            if not refund:
                return self._restricted_reconciliation_result(event, payment_intent_id)
        elif not refund:
            return self._record_event(
                event,
                outcome="ignored",
                reason="Refund is not yet correlated to an ATL request",
            )

        order = self.store.get_order_for_admin(refund["payment_order_id"])
        if (
            not order
            or order["stripe_payment_intent_id"] != payment_intent_id
            or order["currency"] != currency.lower()
            or refund["amount_usd_cents"] != amount
        ):
            return self._record_event(
                event, outcome="rejected", reason="Refund does not match the purchase"
            )

        if not refund.get("stripe_refund_id"):
            refund = self.store.attach_stripe_refund(
                refund["id"], stripe_refund_id=stripe_refund_id
            )
        if refund["stripe_refund_id"] != stripe_refund_id:
            return self._record_event(
                event, outcome="rejected", reason="Stripe Refund does not match"
            )

        if event.event_type == "refund.failed" or status == "failed":
            result = self.store.fail_refund(
                event_id=event.event_id,
                event_type=event.event_type,
                livemode=event.livemode,
                object_id=event.object_id,
                payload_sha256=event.payload_sha256,
                refund_id=refund["id"],
                stripe_refund_id=stripe_refund_id,
            )
        elif status == "succeeded":
            result = self.store.settle_succeeded_refund(
                event_id=event.event_id,
                event_type=event.event_type,
                livemode=event.livemode,
                object_id=event.object_id,
                payload_sha256=event.payload_sha256,
                refund_id=refund["id"],
                stripe_refund_id=stripe_refund_id,
                payment_intent_id=payment_intent_id,
                currency=currency,
                amount_usd_cents=amount,
            )
        else:
            return self._record_event(
                event, outcome="ignored", reason="Refund is awaiting settlement"
            )
        return WebhookResult(
            outcome=result["outcome"],
            event_type=event.event_type,
            reason=result.get("reason"),
            balance_micro=result.get("balance_micro"),
        )

    def _reconcile_external_refund(
        self,
        event: StripeWebhookEvent,
        *,
        payment_intent_id: str,
        amount_usd_cents: int,
        currency: str,
    ) -> dict[str, Any] | None:
        order = self.store.get_order_by_payment_intent(payment_intent_id)
        if not order or order["currency"] != currency.lower():
            return None
        refund_id = _operation_id("recon", event.object_id)
        try:
            return self.store.reserve_reconciliation_refund(
                refund_id=refund_id,
                payment_order_id=order["id"],
                user_id=order["user_id"],
                amount_usd_cents=amount_usd_cents,
                credits_micro=credits_micro_for_cents(amount_usd_cents),
                stripe_refund_id=event.object_id,
            )
        except RefundNotAllowedError:
            self.store.restrict_account(order["user_id"])
            return None

    def _restricted_reconciliation_result(
        self, event: StripeWebhookEvent, payment_intent_id: str
    ) -> WebhookResult:
        order = self.store.get_order_by_payment_intent(payment_intent_id)
        if order:
            self.store.restrict_account(order["user_id"])
        return self._record_event(
            event,
            outcome="rejected",
            reason="Refund requires administrator reconciliation",
            account_restricted=bool(order),
        )


credits_service = CreditsService()
