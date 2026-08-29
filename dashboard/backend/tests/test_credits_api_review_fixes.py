"""HTTP-boundary regressions for the Credits API.

Covers the three things the route layer got wrong: an operator misconfiguration
reported as a client error (or as a bare 500 that reads as CORS), an admin gate
re-implemented instead of reused, and an unauthenticated webhook that buffered
an unbounded body before it could authenticate anything.
"""

from __future__ import annotations

import inspect

from dashboard.backend.domain.credits.config import BillingConfigurationError

from dashboard.backend.tests.test_credits_api import (  # noqa: F401 - fixture
    _auth,
    _promote_admin,
    _signup,
    billing_api,
)


# ---------------------------------------------------------------------------
# Operator misconfiguration is a 503, not a 422 and not a 500.
# ---------------------------------------------------------------------------

def test_balance_reports_bad_billing_config_as_503(billing_api):
    """BillingConfigurationError subclasses ValueError.

    get_balance resolves the config on every call while the gateway singleton
    is unset, so a malformed ATL_STRIPE_TEST_BILLING_ENABLED reached this
    handler as an exception. With no try/except it became a bare 500, which
    escapes CORSMiddleware un-headered and shows up in the browser as a CORS
    failure rather than a server error.
    """
    token = _signup(billing_api.client, "balance-config@example.com")

    def _boom(*_args, **_kwargs):
        raise BillingConfigurationError("enable flag must be true or false")

    billing_api.service.get_balance = _boom
    response = billing_api.client.get("/api/credits/balance", headers=_auth(token))

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_checkout_reports_bad_billing_config_as_503_not_422(billing_api):
    """422 blamed the caller for a request they got right."""
    token = _signup(billing_api.client, "checkout-config@example.com")

    def _boom(*_args, **_kwargs):
        raise BillingConfigurationError("STRIPE_SECRET_KEY must be a Test Mode key")

    billing_api.service.create_checkout = _boom
    response = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "22222222-2222-4222-8222-222222222222",
            "package_id": "usd_5",
        },
    )

    assert response.status_code == 503


def test_ledger_store_error_is_not_a_bare_500(billing_api):
    """The read routes had no error mapping at all."""
    token = _signup(billing_api.client, "ledger-error@example.com")
    billing_api.service.list_ledger = lambda *a, **k: (_ for _ in ()).throw(
        ValueError("limit must be positive")
    )

    response = billing_api.client.get("/api/credits/ledger", headers=_auth(token))
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# One admin gate, not a per-router copy.
# ---------------------------------------------------------------------------

def test_admin_routes_depend_on_the_shared_require_admin(billing_api):
    """api/auth.py documents require_admin as "the one admin gate".

    A local re-implementation means any future change to what counts as an
    admin (a second role, a suspended-admin check, an audit hook) lands in
    require_admin and silently misses the two routes that move money.
    """
    from dashboard.backend.api.auth import require_admin
    from dashboard.backend.api.routers import credits as credits_router

    for handler in (
        credits_router.get_admin_credit_orders,
        credits_router.create_admin_credit_refund,
    ):
        dependencies = [
            param.default.dependency
            for param in inspect.signature(handler).parameters.values()
            if hasattr(param.default, "dependency")
        ]
        assert require_admin in dependencies, (
            f"{handler.__name__} must depend on the shared require_admin gate"
        )

    assert not hasattr(credits_router, "_require_admin"), (
        "the local admin check was replaced by the shared dependency"
    )


def test_non_admin_still_refused_on_admin_orders(billing_api):
    token = _signup(billing_api.client, "not-admin@example.com")
    response = billing_api.client.get(
        "/api/admin/credits/orders", headers=_auth(token)
    )
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# The webhook is the one unauthenticated route here.
# ---------------------------------------------------------------------------

def test_oversized_webhook_body_is_refused(billing_api):
    """Signature verification needs the raw bytes, so the read comes first.

    That is precisely why it must be bounded: without a ceiling one anonymous
    POST can buffer an arbitrary body into a 512MB instance.
    """
    from dashboard.backend.api.routers import credits as credits_router

    oversized = b"x" * (credits_router._MAX_WEBHOOK_BODY_BYTES + 1)
    response = billing_api.client.post(
        "/api/webhooks/stripe",
        content=oversized,
        headers={"Stripe-Signature": "valid-signature"},
    )

    assert response.status_code == 413
    assert billing_api.gateway.webhook_payloads == [], (
        "an oversized body must be refused before it reaches signature verification"
    )


def test_understated_content_length_is_still_refused(billing_api):
    """A hostile client can lie about Content-Length, so the stream is counted."""
    from dashboard.backend.api.routers import credits as credits_router

    limit = credits_router._MAX_WEBHOOK_BODY_BYTES

    def _chunks():
        for _ in range((limit // 1024) + 2):
            yield b"y" * 1024

    response = billing_api.client.post(
        "/api/webhooks/stripe",
        content=_chunks(),
        headers={"Stripe-Signature": "valid-signature"},
    )
    assert response.status_code == 413


def test_normal_sized_webhook_still_reaches_verification(billing_api):
    """The cap must not break the ordinary path."""
    response = billing_api.client.post(
        "/api/webhooks/stripe",
        content=b'{"id": "evt_small"}',
        headers={"Stripe-Signature": "wrong-signature"},
    )
    # 400 = the body was read and handed to signature verification, which
    # rejected it. Anything else would mean the cap swallowed a valid request.
    assert response.status_code == 400
    assert billing_api.gateway.webhook_payloads == [b'{"id": "evt_small"}']


def test_webhook_carries_a_flood_guard(billing_api):
    from dashboard.backend.api.routers import credits as credits_router

    assert credits_router._WEBHOOK_LIMITER.max_events > 0, (
        "the one route here with no authentication in front of it needs the "
        "same flood guard every authenticated spend route already has"
    )


def test_admin_can_reinstate_a_restricted_account(billing_api):
    """Restriction is applied automatically and now actually blocks purchases.

    Without a remedy the only way to clear it would be SQL against production.
    """
    token = _signup(billing_api.client, "restricted-buyer@example.com")
    billing_api.store.restrict_account(1)
    _promote_admin(billing_api.users, 1)

    blocked = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "33333333-3333-4333-8333-333333333333",
            "package_id": "usd_5",
        },
    )
    assert blocked.status_code == 403

    reinstated = billing_api.client.post(
        "/api/admin/credits/accounts/1/reinstate", headers=_auth(token)
    )
    assert reinstated.status_code == 200
    assert reinstated.json()["balance"]["account_status"] == "active"

    allowed = billing_api.client.post(
        "/api/credits/checkout-sessions",
        headers=_auth(token),
        json={
            "client_request_id": "33333333-3333-4333-8333-333333333333",
            "package_id": "usd_5",
        },
    )
    assert allowed.status_code == 200


def test_reinstate_requires_admin(billing_api):
    token = _signup(billing_api.client, "plain-user-reinstate@example.com")
    response = billing_api.client.post(
        "/api/admin/credits/accounts/1/reinstate", headers=_auth(token)
    )
    assert response.status_code == 403
