"""Credits purchase/refund orchestration without Stripe network access."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from uuid import UUID

import pytest
from pydantic import ValidationError

from dashboard.backend.domain.credits.models import (
    AdminRefundRequest,
    CheckoutRequest,
    credits_micro_for_cents,
    format_credits,
)
from dashboard.backend.domain.credits.repository import CreditsStore, OrderConflictError
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.credits.stripe_gateway import (
    CheckoutSessionResult,
    InvalidWebhookSignatureError,
    RefundResult,
    StripeGatewayError,
    StripeWebhookEvent,
)


CLIENT_REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")


def _store(tmp_path) -> CreditsStore:
    path = tmp_path / "credits-service.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO users (id, email, display_name, password_hash, role, created_at)
            VALUES (?, ?, ?, 'unused', ?, '2026-08-13T00:00:00+00:00')
            """,
            [
                (1, "buyer@example.com", "Buyer", "user"),
                (2, "admin@example.com", "Admin", "admin"),
            ],
        )
    return CreditsStore(path)


class FakeGateway:
    def __init__(self):
        self.checkout_calls = []
        self.refund_calls = []
        self.event = None
        self.checkout_error = None
        self.refund_error = None

    def create_checkout_session(self, **kwargs):
        self.checkout_calls.append(kwargs)
        if self.checkout_error:
            raise self.checkout_error
        return CheckoutSessionResult(
            session_id=f"cs_test_{kwargs['order_id']}",
            checkout_url=f"https://checkout.stripe.test/{kwargs['order_id']}",
            payment_intent_id=None,
            payment_status="unpaid",
        )

    def create_refund(self, **kwargs):
        self.refund_calls.append(kwargs)
        if self.refund_error:
            raise self.refund_error
        return RefundResult(
            refund_id=f"re_test_{kwargs['refund_id']}",
            payment_intent_id=kwargs["payment_intent_id"],
            amount_usd_cents=kwargs["amount_usd_cents"],
            status="pending",
        )

    def verify_webhook(self, payload, signature_header):
        if signature_header != "valid":
            raise InvalidWebhookSignatureError("invalid signature")
        return self.event


def _service(tmp_path):
    gateway = FakeGateway()
    return CreditsService(store=_store(tmp_path), gateway=gateway), gateway


def _checkout(service, *, cents=None, request_id=CLIENT_REQUEST_ID):
    request = (
        CheckoutRequest(
            client_request_id=request_id,
            custom_amount_usd_cents=cents,
        )
        if cents is not None
        else CheckoutRequest(
            client_request_id=request_id,
            package_id="usd_10",
        )
    )
    return service.create_checkout(1, request)


def _checkout_event(
    checkout,
    *,
    event_id="evt_checkout_paid",
    payment_status="paid",
    amount_usd_cents=None,
    livemode=False,
):
    return StripeWebhookEvent(
        event_id=event_id,
        event_type="checkout.session.completed",
        livemode=livemode,
        object_id=checkout.checkout_session_id,
        payload_sha256=event_id.ljust(64, "a"),
        data_object={
            "id": checkout.checkout_session_id,
            "client_reference_id": checkout.order_id,
            "payment_status": payment_status,
            "payment_intent": f"pi_test_{checkout.order_id}",
            "amount_total": amount_usd_cents or checkout.amount_usd_cents,
            "currency": "usd",
            "metadata": {
                "atl_order_id": checkout.order_id,
                "atl_user_reference": "1",
                "atl_credits_micro": str(checkout.credits_micro),
            },
        },
    )


def _pay(service, gateway, checkout):
    gateway.event = _checkout_event(checkout)
    result = service.handle_webhook(b"checkout", "valid")
    assert result.outcome == "processed"
    return result


def _refund_event(
    refund,
    checkout,
    *,
    event_id="evt_refund",
    status="succeeded",
    event_type="refund.updated",
    include_local_metadata=True,
    amount_usd_cents=None,
):
    metadata = {"atl_refund_id": refund.refund_id} if include_local_metadata else {}
    return StripeWebhookEvent(
        event_id=event_id,
        event_type=event_type,
        livemode=False,
        object_id=refund.stripe_refund_id,
        payload_sha256=event_id.ljust(64, "b"),
        data_object={
            "id": refund.stripe_refund_id,
            "payment_intent": f"pi_test_{checkout.order_id}",
            "amount": amount_usd_cents or refund.amount_usd_cents,
            "currency": "usd",
            "status": status,
            "metadata": metadata,
        },
    )


@pytest.mark.parametrize(
    ("package_id", "cents"),
    [("usd_5", 500), ("usd_10", 1000), ("usd_20", 2000), ("usd_50", 5000)],
)
def test_fixed_packages_are_resolved_server_side(package_id, cents):
    request = CheckoutRequest(
        client_request_id=CLIENT_REQUEST_ID,
        package_id=package_id,
    )

    assert request.amount_usd_cents == cents
    assert credits_micro_for_cents(cents) == cents * 10_000


@pytest.mark.parametrize("cents", [500, 501, 20_000])
def test_custom_amount_accepts_integer_cent_boundaries(cents):
    request = CheckoutRequest(
        client_request_id=CLIENT_REQUEST_ID,
        custom_amount_usd_cents=cents,
    )

    assert request.amount_usd_cents == cents


@pytest.mark.parametrize("cents", [499, 20_001, 500.0, True])
def test_custom_amount_rejects_out_of_range_or_non_integer_values(cents):
    with pytest.raises(ValidationError):
        CheckoutRequest(
            client_request_id=CLIENT_REQUEST_ID,
            custom_amount_usd_cents=cents,
        )


def test_checkout_requires_exactly_one_price_choice():
    with pytest.raises(ValidationError, match="exactly one"):
        CheckoutRequest(client_request_id=CLIENT_REQUEST_ID)
    with pytest.raises(ValidationError, match="exactly one"):
        CheckoutRequest(
            client_request_id=CLIENT_REQUEST_ID,
            package_id="usd_5",
            custom_amount_usd_cents=500,
        )


def test_balance_format_is_exact_and_not_float_based(tmp_path):
    service, _ = _service(tmp_path)

    assert service.get_balance(1).model_dump() == {
        "balance_micro": 0,
        "display_credits": "0.000000",
    }
    assert format_credits(10_000_001) == "10.000001"
    assert format_credits(-4_000_000) == "-4.000000"


def test_checkout_retry_reuses_order_and_stripe_idempotency_key(tmp_path):
    service, gateway = _service(tmp_path)

    first = _checkout(service)
    retried = _checkout(service)

    assert retried == first
    assert len(gateway.checkout_calls) == 2
    assert gateway.checkout_calls[0]["order_id"] == first.order_id
    assert (
        gateway.checkout_calls[0]["idempotency_key"]
        == (gateway.checkout_calls[1]["idempotency_key"])
        == f"checkout:{first.order_id}"
    )
    assert service.store.get_balance_micro(1) == 0


def test_same_client_request_cannot_change_the_purchase_amount(tmp_path):
    service, _ = _service(tmp_path)
    _checkout(service)

    with pytest.raises(OrderConflictError, match="different purchase"):
        _checkout(service, cents=500)


def test_checkout_gateway_timeout_leaves_one_retryable_pending_order(tmp_path):
    service, gateway = _service(tmp_path)
    gateway.checkout_error = StripeGatewayError("timeout")

    with pytest.raises(StripeGatewayError):
        _checkout(service)
    gateway.checkout_error = None
    completed_retry = _checkout(service)

    order = service.store.get_order_for_user(completed_retry.order_id, 1)
    assert order["status"] == "pending"
    assert order["stripe_checkout_session_id"] == completed_retry.checkout_session_id


def test_paid_checkout_posts_once_and_duplicate_events_are_noops(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    gateway.event = _checkout_event(checkout)

    first = service.handle_webhook(b"paid", "valid")
    duplicate = service.handle_webhook(b"paid", "valid")
    gateway.event = _checkout_event(checkout, event_id="evt_checkout_paid_again")
    second_event = service.handle_webhook(b"paid-again", "valid")

    assert first.balance_micro == 10_000_000
    assert duplicate.outcome == second_event.outcome == "duplicate"
    assert service.get_balance(1).balance_micro == 10_000_000
    assert len(service.store.list_ledger_entries(1)["items"]) == 1


@pytest.mark.parametrize(
    ("change", "expected_outcome"),
    [("unpaid", "ignored"), ("amount", "rejected"), ("user", "rejected")],
)
def test_unpaid_or_tampered_checkout_never_credits(tmp_path, change, expected_outcome):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    event = _checkout_event(checkout)
    if change == "unpaid":
        event = replace(
            event,
            data_object={**event.data_object, "payment_status": "unpaid"},
        )
    elif change == "amount":
        event = replace(
            event,
            data_object={**event.data_object, "amount_total": 999},
        )
    else:
        metadata = {**event.data_object["metadata"], "atl_user_reference": "2"}
        event = replace(
            event,
            data_object={**event.data_object, "metadata": metadata},
        )
    gateway.event = event

    result = service.handle_webhook(b"tampered", "valid")

    assert result.outcome == expected_outcome
    assert service.get_balance(1).balance_micro == 0


def test_invalid_signature_and_unsupported_event_are_safe(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    gateway.event = replace(
        _checkout_event(checkout),
        event_id="evt_unsupported",
        event_type="customer.created",
    )

    with pytest.raises(InvalidWebhookSignatureError):
        service.handle_webhook(b"forged", "invalid")
    ignored = service.handle_webhook(b"unsupported", "valid")

    assert ignored.outcome == "ignored"
    assert service.get_balance(1).balance_micro == 0


def test_even_unsupported_live_mode_events_are_rejected(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    gateway.event = replace(
        _checkout_event(checkout),
        event_id="evt_live_unsupported",
        event_type="customer.created",
        livemode=True,
    )

    result = service.handle_webhook(b"live-unsupported", "valid")

    assert result.outcome == "rejected"
    assert "Live Mode" in result.reason
    assert service.get_balance(1).balance_micro == 0


def test_admin_refund_reserves_calls_stripe_and_settles_negative_entry(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)

    request = AdminRefundRequest(
        client_request_id=UUID("22222222-2222-4222-8222-222222222222"),
        payment_order_id=checkout.order_id,
        amount_usd_cents=400,
    )
    refund = service.create_admin_refund(2, request)

    assert refund.refund_status == "submitted"
    assert gateway.refund_calls[0]["idempotency_key"] == f"refund:{refund.refund_id}"
    gateway.event = _refund_event(refund, checkout)
    settled = service.handle_webhook(b"refund", "valid")

    assert settled.outcome == "processed"
    assert settled.balance_micro == 6_000_000
    assert service.store.get_order_for_admin(checkout.order_id)["status"] == (
        "partially_refunded"
    )


def test_duplicate_refund_events_never_reverse_credits_twice(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    refund = service.create_admin_refund(
        2,
        AdminRefundRequest(
            client_request_id=UUID("23232323-2323-4232-8232-232323232323"),
            payment_order_id=checkout.order_id,
            amount_usd_cents=400,
        ),
    )
    gateway.event = _refund_event(refund, checkout)

    first = service.handle_webhook(b"refund", "valid")
    duplicate = service.handle_webhook(b"refund", "valid")
    gateway.event = _refund_event(
        refund,
        checkout,
        event_id="evt_refund_replayed_as_new_event",
    )
    second_event = service.handle_webhook(b"refund-again", "valid")

    assert first.balance_micro == 6_000_000
    assert duplicate.outcome == second_event.outcome == "duplicate"
    assert service.get_balance(1).balance_micro == 6_000_000
    assert len(service.store.list_ledger_entries(1)["items"]) == 2


def test_pending_refund_event_records_no_credit_change_then_success_can_settle(
    tmp_path,
):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    refund = service.create_admin_refund(
        2,
        AdminRefundRequest(
            client_request_id=UUID("24242424-2424-4242-8242-242424242424"),
            payment_order_id=checkout.order_id,
            amount_usd_cents=400,
        ),
    )
    gateway.event = _refund_event(
        refund,
        checkout,
        event_id="evt_refund_pending",
        status="pending",
        event_type="refund.created",
    )

    pending = service.handle_webhook(b"pending", "valid")
    balance_while_pending = service.get_balance(1).balance_micro
    gateway.event = _refund_event(
        refund,
        checkout,
        event_id="evt_refund_later_succeeded",
    )
    succeeded = service.handle_webhook(b"succeeded", "valid")

    assert pending.outcome == "ignored"
    assert balance_while_pending == 10_000_000
    assert succeeded.balance_micro == 6_000_000


def test_live_mode_refund_is_rejected_without_balance_change(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    refund = service.create_admin_refund(
        2,
        AdminRefundRequest(
            client_request_id=UUID("25252525-2525-4252-8252-252525252525"),
            payment_order_id=checkout.order_id,
            amount_usd_cents=400,
        ),
    )
    gateway.event = replace(_refund_event(refund, checkout), livemode=True)

    result = service.handle_webhook(b"live-refund", "valid")

    assert result.outcome == "rejected"
    assert service.get_balance(1).balance_micro == 10_000_000


def test_refund_gateway_timeout_keeps_one_retryable_reservation(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    request = AdminRefundRequest(
        client_request_id=UUID("33333333-3333-4333-8333-333333333333"),
        payment_order_id=checkout.order_id,
        amount_usd_cents=400,
    )
    gateway.refund_error = StripeGatewayError("timeout")

    with pytest.raises(StripeGatewayError):
        service.create_admin_refund(2, request)
    gateway.refund_error = None
    retried = service.create_admin_refund(2, request)

    assert retried.refund_status == "submitted"
    assert gateway.refund_calls[0]["idempotency_key"] == (
        gateway.refund_calls[1]["idempotency_key"]
    )
    assert (
        service.store.list_orders_for_admin()["items"][0]["refundable_usd_cents"] == 600
    )


def test_failed_refund_event_releases_reservation(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    request = AdminRefundRequest(
        client_request_id=UUID("44444444-4444-4444-8444-444444444444"),
        payment_order_id=checkout.order_id,
        amount_usd_cents=1000,
    )
    refund = service.create_admin_refund(2, request)
    gateway.event = _refund_event(
        refund,
        checkout,
        status="failed",
        event_type="refund.failed",
    )

    failed = service.handle_webhook(b"failed", "valid")

    assert failed.outcome == "processed"
    assert (
        service.store.list_orders_for_admin()["items"][0]["refundable_usd_cents"]
        == 1000
    )


def test_succeeded_dashboard_refund_is_reconciled_without_local_metadata(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    external = RefundCreationResultFixture(
        refund_id="unused",
        stripe_refund_id="re_test_external",
        amount_usd_cents=400,
    )
    gateway.event = _refund_event(
        external,
        checkout,
        event_id="evt_external_refund",
        include_local_metadata=False,
    )

    reconciled = service.handle_webhook(b"external", "valid")

    assert reconciled.outcome == "processed"
    assert reconciled.balance_micro == 6_000_000
    ledger = service.store.list_ledger_entries(1)["items"]
    assert sorted(entry["amount_micro"] for entry in ledger) == [
        -4_000_000,
        10_000_000,
    ]


def test_unaffordable_dashboard_refund_restricts_account(tmp_path):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    external = RefundCreationResultFixture(
        refund_id="unused",
        stripe_refund_id="re_test_too_large",
        amount_usd_cents=1100,
    )
    gateway.event = _refund_event(
        external,
        checkout,
        event_id="evt_external_too_large",
        include_local_metadata=False,
    )

    result = service.handle_webhook(b"too-large", "valid")

    assert result.outcome == "rejected"
    assert result.account_restricted is True
    admin_order = service.store.list_orders_for_admin()["items"][0]
    assert admin_order["account_status"] == "restricted"
    assert service.get_balance(1).balance_micro == 10_000_000


class RefundCreationResultFixture:
    def __init__(self, *, refund_id, stripe_refund_id, amount_usd_cents):
        self.refund_id = refund_id
        self.stripe_refund_id = stripe_refund_id
        self.amount_usd_cents = amount_usd_cents
