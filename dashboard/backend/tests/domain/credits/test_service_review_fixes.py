"""Regressions for the money-losing and money-hiding paths in the Credits service.

Each case here pins a defect where the customer was charged and the failure was
invisible, or where a transient failure permanently consumed an order's
refundable amount. They share the fixtures in ``test_service.py`` rather than
rebuilding the harness.
"""

from __future__ import annotations

from dataclasses import replace
from uuid import UUID

import pytest

from dashboard.backend.domain.credits.models import AdminRefundRequest
from dashboard.backend.domain.credits.service import AccountRestrictedError
from dashboard.backend.domain.credits.stripe_gateway import (
    StripeGatewayDefinitiveError,
    StripeGatewayError,
    StripeWebhookEvent,
)

from dashboard.backend.tests.domain.credits.test_service import (
    _checkout,
    _checkout_event,
    _pay,
    _service,
    _store,
)


REFUND_REQUEST_ID = UUID("44444444-4444-4444-8444-444444444444")


# ---------------------------------------------------------------------------
# The purchase gate is not the browser.
# ---------------------------------------------------------------------------

def test_restricted_account_cannot_create_a_checkout_session(tmp_path):
    """credits.js disables the buttons; that is a hint, not a gate.

    An account restricted by an over-refund reconciliation could still POST
    /api/credits/checkout-sessions directly and keep buying.
    """
    service, _gateway = _service(tmp_path)
    service.store.restrict_account(1)

    with pytest.raises(AccountRestrictedError):
        _checkout(service)


def test_active_account_still_creates_a_checkout_session(tmp_path):
    service, gateway = _service(tmp_path)
    assert _checkout(service).order_id
    assert len(gateway.checkout_calls) == 1


# ---------------------------------------------------------------------------
# A rejected webhook must not be silent. Stripe sees 200 either way.
# ---------------------------------------------------------------------------

def test_rejected_webhook_is_logged(tmp_path, capsys):
    """Wholesale upstream drift must not look identical to "no events yet".

    A renamed Stripe field rejects *every* payment while customers are charged.
    The route answers 200, Stripe's dashboard says delivered, and the only other
    record is a table nobody queries.
    """
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    event = _checkout_event(checkout)
    # Simulate an upstream rename: the field the adapter reads is gone.
    broken = dict(event.data_object)
    broken.pop("amount_total")
    gateway.event = replace(event, data_object=broken)

    result = service.handle_webhook(b"checkout", "valid")

    assert result.outcome == "rejected"
    out = capsys.readouterr().out
    assert "ERROR" in out and "rejected" in out
    assert "checkout.session.completed" in out


def test_settled_webhook_does_not_log_noise(tmp_path, capsys):
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    gateway.event = _checkout_event(checkout)
    capsys.readouterr()

    service.handle_webhook(b"checkout", "valid")

    assert "[credits]" not in capsys.readouterr().out


def test_unsupported_event_type_is_logged(tmp_path, capsys):
    service, gateway = _service(tmp_path)
    gateway.event = StripeWebhookEvent(
        event_id="evt_unknown",
        event_type="invoice.paid",
        livemode=False,
        object_id="in_test",
        payload_sha256="e".ljust(64, "e"),
        data_object={"id": "in_test"},
    )

    assert service.handle_webhook(b"x", "valid").outcome == "ignored"
    assert "invoice.paid" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Delayed settlement: the failure twin was handled, the success twin was not.
# ---------------------------------------------------------------------------

def test_async_payment_succeeded_credits_the_account(tmp_path):
    """checkout.session.completed arrives 'unpaid' for delayed methods.

    The settlement then arrives as async_payment_succeeded. Unhandled, it fell
    through to "Unsupported Stripe event type" and the customer was charged and
    never credited.
    """
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)

    gateway.event = _checkout_event(checkout, payment_status="unpaid")
    assert service.handle_webhook(b"pending", "valid").outcome == "ignored"
    assert service.get_balance(1).balance_micro == 0

    settled = _checkout_event(checkout, event_id="evt_async_paid")
    gateway.event = replace(
        settled, event_type="checkout.session.async_payment_succeeded"
    )
    result = service.handle_webhook(b"settled", "valid")

    assert result.outcome == "processed"
    assert service.get_balance(1).balance_micro == checkout.credits_micro


# ---------------------------------------------------------------------------
# The gateway/attach window: money in, no Credits, no recovery.
# ---------------------------------------------------------------------------

def test_checkout_survives_a_failed_session_attach(tmp_path, capsys):
    """Stripe has a live payable session before attach_checkout_session runs.

    A crash or error in that window left stripe_checkout_session_id NULL, which
    the settle path then read as a mismatch and rejected forever.
    """
    service, gateway = _service(tmp_path)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("database went away")

    service.store.attach_checkout_session = _boom
    checkout = _checkout(service)

    assert checkout.checkout_url
    assert "ERROR" in capsys.readouterr().out

    # The signed webhook must still settle the purchase.
    gateway.event = _checkout_event(checkout)
    result = service.handle_webhook(b"checkout", "valid")

    assert result.outcome == "processed"
    assert service.get_balance(1).balance_micro == checkout.credits_micro


def test_settled_orphan_order_records_its_session_id(tmp_path):
    service, gateway = _service(tmp_path)
    service.store.attach_checkout_session = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("nope")
    )
    checkout = _checkout(service)
    gateway.event = _checkout_event(checkout)
    service.handle_webhook(b"checkout", "valid")

    order = service.store.get_order_for_admin(checkout.order_id)
    assert order["stripe_checkout_session_id"] == checkout.checkout_session_id


# ---------------------------------------------------------------------------
# Refund reservations: release only what we know did not happen.
# ---------------------------------------------------------------------------

def _refund_request(order_id, cents, request_id=REFUND_REQUEST_ID):
    return AdminRefundRequest(
        client_request_id=request_id,
        payment_order_id=order_id,
        amount_usd_cents=cents,
    )


def test_definitive_stripe_refusal_releases_the_reservation(tmp_path):
    """Stripe received and refused the request, so no Refund exists.

    Holding the reservation would subtract from the refundable lot forever.
    """
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    gateway.refund_error = StripeGatewayDefinitiveError("refused")

    with pytest.raises(StripeGatewayDefinitiveError):
        service.create_admin_refund(2, _refund_request(checkout.order_id, 400))

    orders = service.store.list_orders_for_admin()["items"]
    assert orders[0]["refundable_usd_cents"] == checkout.amount_usd_cents


def test_ambiguous_stripe_failure_keeps_the_reservation(tmp_path):
    """A timeout may still have created the Refund.

    Releasing here would let the same money be refunded twice, so the
    reservation is held and retried under the same idempotency key instead.
    """
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    gateway.refund_error = StripeGatewayError("timeout")

    with pytest.raises(StripeGatewayError):
        service.create_admin_refund(2, _refund_request(checkout.order_id, 400))

    orders = service.store.list_orders_for_admin()["items"]
    assert orders[0]["refundable_usd_cents"] == checkout.amount_usd_cents - 400


def test_released_reservation_frees_a_full_refund_retry(tmp_path):
    """The whole point: a refused full refund must still be retryable."""
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    full = checkout.amount_usd_cents

    gateway.refund_error = StripeGatewayDefinitiveError("refused")
    with pytest.raises(StripeGatewayDefinitiveError):
        service.create_admin_refund(2, _refund_request(checkout.order_id, full))

    gateway.refund_error = None
    retried = service.create_admin_refund(
        2,
        _refund_request(
            checkout.order_id, full, UUID("55555555-5555-4555-8555-555555555555")
        ),
    )
    assert retried.refund_status == "submitted"


def test_cancelled_reservation_is_not_resurrected(tmp_path):
    """Cancelling is one-way for a row that already reached Stripe."""
    service, gateway = _service(tmp_path)
    checkout = _checkout(service)
    _pay(service, gateway, checkout)
    reserved = service.create_admin_refund(2, _refund_request(checkout.order_id, 400))

    # 'submitted' — a Stripe Refund exists, so the lot must stay reserved.
    assert service.store.cancel_refund_reservation(reserved.refund_id)["status"] == (
        "submitted"
    )
    orders = service.store.list_orders_for_admin()["items"]
    assert orders[0]["refundable_usd_cents"] == checkout.amount_usd_cents - 400


def test_cancel_reservation_is_a_noop_for_an_unknown_id(tmp_path):
    service, _gateway = _service(tmp_path)
    assert service.store.cancel_refund_reservation("rfnd_missing") is None


# ---------------------------------------------------------------------------
# Connection hygiene.
# ---------------------------------------------------------------------------

def test_store_closes_every_connection_it_opens(tmp_path):
    """sqlite3's `with conn` commits; it never closes.

    Left as-is each call leaked a file handle and, in WAL mode, a read mark
    that defers checkpointing until CPython happens to collect it.
    """
    store = _store(tmp_path)
    opened: list = []
    real_connect = __import__("sqlite3").connect

    import sqlite3 as _sqlite3

    def _tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    _sqlite3.connect = _tracking_connect
    try:
        store.ensure_account(1)
        store.get_balance_micro(1)
        store.list_ledger_entries(1, limit=10, cursor=None)
    finally:
        _sqlite3.connect = real_connect

    assert opened, "expected the store to open at least one connection"
    for conn in opened:
        with pytest.raises(_sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


def test_store_still_rolls_back_on_error(tmp_path):
    """Closing must not cost the transaction semantics `with conn` provided."""
    service, _gateway = _service(tmp_path)
    checkout = _checkout(service)
    before = service.store.get_order_for_admin(checkout.order_id)

    with pytest.raises(Exception):
        service.store.reserve_refund(
            refund_id="rfnd_bad",
            payment_order_id=checkout.order_id,
            user_id=1,
            requested_by_user_id=2,
            amount_usd_cents=999_999,
            credits_micro=999_999_000_000,
        )

    after = service.store.get_order_for_admin(checkout.order_id)
    assert after["status"] == before["status"]
