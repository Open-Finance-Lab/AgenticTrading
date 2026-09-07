"""Checkout intent instrumentation contracts."""

from __future__ import annotations

from dashboard.backend.api.routers import credits as credits_router
from dashboard.backend.domain.analytics import instrumentation
from dashboard.backend.domain.analytics.models import AppendEventResult
from dashboard.backend.domain.credits.models import CheckoutRequest, CheckoutResult


class CheckoutService:
    def create_checkout(self, user_id, request):
        assert user_id == 7
        assert request.package_id == "usd_5"
        return CheckoutResult(
            order_id="ord-synthetic-7",
            checkout_session_id="cs-synthetic-7",
            checkout_url="https://checkout.example.test/synthetic",
            amount_usd_cents=500,
            credits_micro=5_000_000,
            order_status="pending",
        )


class RecordingAnalyticsService:
    project_snapshots = True

    def __init__(self):
        self.calls = []

    def try_record_server_event(self, **kwargs):
        self.calls.append(kwargs)
        return AppendEventResult.model_construct(event=None, created=True)


def _request():
    return CheckoutRequest(
        client_request_id="11111111-1111-4111-8111-111111111111",
        package_id="usd_5",
    )


def test_checkout_route_emits_once_after_order_creation(monkeypatch):
    events = []
    monkeypatch.setattr(credits_router, "credits_service", CheckoutService())
    monkeypatch.setattr(
        credits_router.analytics_instrumentation,
        "emit_resource_event",
        lambda **kwargs: events.append(kwargs),
    )
    credits_router._CHECKOUT_LIMITER.reset()

    response = credits_router.create_credit_checkout(_request(), {"id": 7})

    assert response["checkout"]["order_id"] == "ord-synthetic-7"
    assert events == [
        {
            "event_name": "checkout_started",
            "user_id": 7,
            "source_record_type": "credit_checkout",
            "source_record_id": "ord-synthetic-7",
            "properties": {},
        }
    ]


def test_checkout_analytics_failure_does_not_fail_checkout(monkeypatch):
    monkeypatch.setattr(credits_router, "credits_service", CheckoutService())
    monkeypatch.setattr(
        credits_router.analytics_instrumentation,
        "emit_resource_event",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    credits_router._CHECKOUT_LIMITER.reset()

    response = credits_router.create_credit_checkout(_request(), {"id": 7})

    assert response["checkout"]["order_id"] == "ord-synthetic-7"


def test_checkout_emitter_uses_stable_idempotency_reference(monkeypatch):
    service = RecordingAnalyticsService()
    monkeypatch.setattr(instrumentation, "get_analytics_service", lambda: service)

    for _ in range(2):
        instrumentation.emit_resource_event(
            event_name="checkout_started",
            user_id=7,
            source_record_type="credit_checkout",
            source_record_id="ord-synthetic-7",
            properties={},
        )

    assert [call["source_event_id"] for call in service.calls] == [
        "resource:checkout_started:ord-synthetic-7",
        "resource:checkout_started:ord-synthetic-7",
    ]
