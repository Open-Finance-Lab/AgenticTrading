"""Authenticated Credits APIs and the signed Stripe webhook boundary."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from dashboard.backend.api.auth import get_current_user, require_admin
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter, client_key
from dashboard.backend.domain.credits.config import (
    BillingConfigurationError,
    BillingUnavailableError,
)
from dashboard.backend.domain.credits.models import (
    AdminRefundRequest,
    CheckoutRequest,
    format_credits,
)
from dashboard.backend.domain.credits.repository import (
    OrderConflictError,
    RefundNotAllowedError,
)
from dashboard.backend.domain.credits.service import (
    AccountRestrictedError,
    CreditsServiceError,
    PaymentOrderNotFoundError,
    credits_service,
)
from dashboard.backend.domain.credits.stripe_gateway import (
    InvalidWebhookSignatureError,
    StripeGatewayError,
)


router = APIRouter(tags=["credits"])

# Per-user, in-process abuse guards. Authentication remains the security
# boundary; these limits cap accidental duplicate Stripe work on one worker.
_CHECKOUT_LIMITER = FixedWindowRateLimiter(max_events=10, window_seconds=60)
_ORDER_POLL_LIMITER = FixedWindowRateLimiter(max_events=120, window_seconds=60)
_ADMIN_REFUND_LIMITER = FixedWindowRateLimiter(max_events=20, window_seconds=300)
# The webhook is the one route here with no authentication in front of it, so
# it gets the flood guard every other spend-touching route already had. Budget
# is deliberately far above real Stripe volume for this app: a 429 costs
# nothing (Stripe retries, and every handler below is idempotent), but a tight
# budget would delay a legitimate settlement burst.
_WEBHOOK_LIMITER = FixedWindowRateLimiter(max_events=600, window_seconds=60)

# Signature verification needs the raw bytes, so the body must be read before
# the request can be authenticated at all — which is exactly why the read has
# to be bounded. A Stripe event is a few KB; 256 KiB is generous. Without this
# a single anonymous POST can buffer an arbitrary body into a 512MB instance.
_MAX_WEBHOOK_BODY_BYTES = 256 * 1024


def _rate_limit(limiter: FixedWindowRateLimiter, *, key: str, detail: str) -> None:
    if limiter.allow(key):
        return
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(limiter.retry_after_seconds(key))},
    )


def _public_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": order["id"],
        "status": order["status"],
        "currency": order["currency"],
        "amount_usd_cents": order["amount_usd_cents"],
        "credits_micro": order["credits_micro"],
        "display_credits": format_credits(order["credits_micro"]),
        "created_at": order["created_at"],
        "updated_at": order["updated_at"],
        "paid_at": order["paid_at"],
    }


def _public_ledger_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = {
        "id": entry["id"],
        "source_kind": entry.get("source_kind", "ledger"),
        "bucket": entry["bucket"],
        "entry_type": entry["entry_type"],
        "amount_micro": entry["amount_micro"],
        "display_credits": format_credits(entry["amount_micro"]),
        "source": entry["source"],
        "reason": entry["reason"],
        "payment_order_id": entry["payment_order_id"],
        "created_at": entry["created_at"],
    }
    if entry.get("entry_type") == "backtest_usage":
        result.update(
            {
                "run_id": entry.get("run_id"),
                "model_call_count": entry.get("model_call_count"),
                "provider_id": entry.get("provider_id"),
                "model_id": entry.get("model_id"),
                "provider_mixed": bool(entry.get("provider_mixed")),
                "model_mixed": bool(entry.get("model_mixed")),
                "billing_source": entry.get("billing_source"),
            }
        )
    return result


def _public_admin_order(order: dict[str, Any]) -> dict[str, Any]:
    return {
        "sequence": order["sequence"],
        "order_id": order["id"],
        "user_id": order["user_id"],
        "status": order["status"],
        "account_status": order["account_status"],
        "currency": order["currency"],
        "amount_usd_cents": order["amount_usd_cents"],
        "credits_micro": order["credits_micro"],
        "refundable_usd_cents": order["refundable_usd_cents"],
        "refundable_credits_micro": order["refundable_credits_micro"],
        "created_at": order["created_at"],
        "updated_at": order["updated_at"],
        "paid_at": order["paid_at"],
    }


def _raise_billing_http_error(exc: Exception) -> None:
    if isinstance(exc, BillingUnavailableError):
        raise HTTPException(
            status_code=503, detail="Stripe Test Mode billing is unavailable"
        ) from exc
    # Must precede the ValueError arm below: BillingConfigurationError subclasses
    # ValueError, so it would otherwise report an *operator* misconfiguration
    # (a malformed enable flag, a live-mode key) as a 422 blaming the caller for
    # a request they got right. It is the same class of failure as
    # BillingUnavailableError and gets the same 503.
    if isinstance(exc, BillingConfigurationError):
        raise HTTPException(
            status_code=503, detail="Stripe Test Mode billing is unavailable"
        ) from exc
    if isinstance(exc, PaymentOrderNotFoundError):
        raise HTTPException(status_code=404, detail=exc.message) from exc
    # Before the generic CreditsServiceError arm below, which would report this
    # as a 409 conflict rather than a refusal.
    if isinstance(exc, AccountRestrictedError):
        raise HTTPException(status_code=403, detail=exc.message) from exc
    if isinstance(exc, RefundNotAllowedError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (OrderConflictError, CreditsServiceError)):
        detail = exc.message if isinstance(exc, CreditsServiceError) else str(exc)
        raise HTTPException(status_code=409, detail=detail) from exc
    if isinstance(exc, StripeGatewayError):
        raise HTTPException(
            status_code=503, detail="Stripe Test Mode operation failed"
        ) from exc
    if isinstance(exc, (ValueError, KeyError)):
        raise HTTPException(status_code=422, detail="Invalid billing request") from exc
    raise exc


@router.get("/credits/balance")
def get_credit_balance(current_user: dict = Depends(get_current_user)):
    # get_balance resolves the billing config on every call while the gateway
    # singleton is unset, so a malformed operator flag reaches this handler as
    # an exception. Unwrapped it became a bare 500, which escapes CORSMiddleware
    # un-headered and reads in the browser as a CORS failure.
    try:
        result = credits_service.get_balance(int(current_user["id"]))
    except Exception as exc:
        _raise_billing_http_error(exc)
    return {"balance": result.model_dump(), "test_mode": True}


@router.get("/credits/ledger")
def get_credit_ledger(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, min_length=1, max_length=256),
    current_user: dict = Depends(get_current_user),
):
    try:
        page = credits_service.list_ledger(
            int(current_user["id"]), limit=limit, cursor=cursor
        )
    except Exception as exc:
        _raise_billing_http_error(exc)
    return {
        "items": [_public_ledger_entry(item) for item in page["items"]],
        "next_cursor": page["next_cursor"],
    }


@router.post("/credits/checkout-sessions")
def create_credit_checkout(
    payload: CheckoutRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["id"])
    _rate_limit(
        _CHECKOUT_LIMITER,
        key=f"checkout:{user_id}",
        detail="Too many checkout requests; please try again later",
    )
    try:
        result = credits_service.create_checkout(user_id, payload)
    except Exception as exc:
        _raise_billing_http_error(exc)
    return {"checkout": result.model_dump(), "test_mode": True}


@router.get("/credits/orders/{order_id}")
def get_credit_order(
    order_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = int(current_user["id"])
    _rate_limit(
        _ORDER_POLL_LIMITER,
        key=f"order-poll:{user_id}",
        detail="Too many order status requests; please try again later",
    )
    try:
        order = credits_service.get_order(order_id, user_id)
    except Exception as exc:
        _raise_billing_http_error(exc)
    if not order:
        raise HTTPException(status_code=404, detail="Payment order was not found")
    return {"order": _public_order(order), "test_mode": True}


@router.get("/admin/credits/orders")
def get_admin_credit_orders(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int | None = Query(default=None, ge=1),
    _admin: dict = Depends(require_admin),
):
    try:
        page = credits_service.list_admin_orders(limit=limit, cursor=cursor)
    except Exception as exc:
        _raise_billing_http_error(exc)
    return {
        "items": [_public_admin_order(item) for item in page["items"]],
        "next_cursor": page["next_cursor"],
        "test_mode": True,
    }


@router.post("/admin/credits/refunds")
def create_admin_credit_refund(
    payload: AdminRefundRequest,
    current_user: dict = Depends(require_admin),
):
    admin_id = int(current_user["id"])
    _rate_limit(
        _ADMIN_REFUND_LIMITER,
        key=f"admin-refund:{admin_id}",
        detail="Too many refund requests; please try again later",
    )
    try:
        result = credits_service.create_admin_refund(admin_id, payload)
    except Exception as exc:
        _raise_billing_http_error(exc)
    return {"refund": result.model_dump(), "test_mode": True}


async def _read_bounded_body(request: Request, limit: int) -> bytes:
    """Buffer the request body, refusing anything past ``limit``.

    Checks the declared Content-Length first so an oversized body is refused
    before a byte is read, then counts while streaming — a chunked request can
    omit Content-Length entirely, and a hostile one can understate it.
    """
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_length = int(declared)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length")
        if declared_length > limit:
            raise HTTPException(status_code=413, detail="Stripe payload is too large")

    chunks: list[bytes] = []
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > limit:
            raise HTTPException(status_code=413, detail="Stripe payload is too large")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/admin/credits/accounts/{user_id}/reinstate")
def reinstate_credit_account(
    user_id: int,
    _admin: dict = Depends(require_admin),
):
    """Clear a restriction written by refund reconciliation.

    Restriction is applied automatically (an out-of-band Stripe refund larger
    than the order's refundable amount) and now actually blocks purchases, so
    without this the only remedy is hand-written SQL against production.
    """
    if user_id < 1:
        raise HTTPException(status_code=422, detail="Invalid user id")
    try:
        result = credits_service.reinstate_account(user_id)
    except Exception as exc:
        _raise_billing_http_error(exc)
    return {"balance": result.model_dump(), "test_mode": True}


@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
):
    _rate_limit(
        _WEBHOOK_LIMITER,
        key=client_key(request),
        detail="Too many Stripe webhook deliveries; please retry",
    )
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Stripe signature is required")
    payload = await _read_bounded_body(request, _MAX_WEBHOOK_BODY_BYTES)
    try:
        result = await run_in_threadpool(
            credits_service.handle_webhook,
            payload,
            stripe_signature,
        )
    except InvalidWebhookSignatureError as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature") from exc
    except BillingUnavailableError as exc:
        raise HTTPException(
            status_code=503, detail="Stripe Test Mode billing is unavailable"
        ) from exc
    except (OrderConflictError, StripeGatewayError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Stripe event could not be processed"
        ) from exc
    return {"received": True, "result": result.model_dump()}
