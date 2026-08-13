"""Authenticated Credits APIs and the signed Stripe webhook boundary."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from dashboard.backend.api.auth import get_current_user
from dashboard.backend.api.rate_limit import FixedWindowRateLimiter
from dashboard.backend.domain.credits.config import BillingUnavailableError
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


def _rate_limit(limiter: FixedWindowRateLimiter, *, key: str, detail: str) -> None:
    if limiter.allow(key):
        return
    raise HTTPException(
        status_code=429,
        detail=detail,
        headers={"Retry-After": str(limiter.retry_after_seconds(key))},
    )


def _require_admin(current_user: dict) -> None:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")


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
    return {
        "id": entry["id"],
        "entry_type": entry["entry_type"],
        "amount_micro": entry["amount_micro"],
        "display_credits": format_credits(entry["amount_micro"]),
        "payment_order_id": entry["payment_order_id"],
        "created_at": entry["created_at"],
    }


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
    if isinstance(exc, PaymentOrderNotFoundError):
        raise HTTPException(status_code=404, detail=exc.message) from exc
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
    result = credits_service.get_balance(int(current_user["id"]))
    return {"balance": result.model_dump(), "test_mode": True}


@router.get("/credits/ledger")
def get_credit_ledger(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int | None = Query(default=None, ge=1),
    current_user: dict = Depends(get_current_user),
):
    page = credits_service.store.list_ledger_entries(
        int(current_user["id"]), limit=limit, cursor=cursor
    )
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
    order = credits_service.store.get_order_for_user(order_id, user_id)
    if not order:
        raise HTTPException(status_code=404, detail="Payment order was not found")
    return {"order": _public_order(order), "test_mode": True}


@router.get("/admin/credits/orders")
def get_admin_credit_orders(
    limit: int = Query(default=50, ge=1, le=100),
    cursor: int | None = Query(default=None, ge=1),
    current_user: dict = Depends(get_current_user),
):
    _require_admin(current_user)
    page = credits_service.store.list_orders_for_admin(limit=limit, cursor=cursor)
    return {
        "items": [_public_admin_order(item) for item in page["items"]],
        "next_cursor": page["next_cursor"],
        "test_mode": True,
    }


@router.post("/admin/credits/refunds")
def create_admin_credit_refund(
    payload: AdminRefundRequest,
    current_user: dict = Depends(get_current_user),
):
    _require_admin(current_user)
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


@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="Stripe-Signature"),
):
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Stripe signature is required")
    payload = await request.body()
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
