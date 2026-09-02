"""Orchestrate Credit purchases, signed Stripe events, and refunds."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from dashboard.backend.domain.credits.config import load_billing_config
from dashboard.backend.domain.credits.models import (
    AdminRefundRequest,
    BalanceResult,
    BalanceProjection,
    CheckoutRequest,
    CheckoutResult,
    CreditRecoveryResult,
    GrantMutationResult,
    GrantPoolSummary,
    LLMReservation,
    LLMSettlementResult,
    RefundCreationResult,
    WebhookResult,
    credits_micro_for_cents,
    format_credits,
)
from dashboard.backend.domain.credits.repository import (
    OrderConflictError,
    RefundNotAllowedError,
    credits_store,
)
from dashboard.backend.domain.credits.repository_common import _canonical_digest
from dashboard.backend.domain.model_providers.repository_common import (
    validate_provider_id,
)
from dashboard.backend.domain.credits.stripe_gateway import (
    StripeGatewayDefinitiveError,
    StripeGatewayError,
    StripeTestGateway,
    StripeWebhookEvent,
)
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation


DEFAULT_SIGNUP_CREDIT_CAMPAIGN = "default_signup_credits_v1"
DEFAULT_SIGNUP_CREDITS_MICRO = 1_500_000
DEFAULT_SIGNUP_CREDIT_SOURCE = "system_promotion"
DEFAULT_SIGNUP_CREDIT_REASON = "Automatic welcome Credits."


class CreditsServiceError(RuntimeError):
    """A sanitized, expected billing-domain failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class PaymentOrderNotFoundError(CreditsServiceError):
    def __init__(self):
        super().__init__("payment_order_not_found", "Payment order was not found")


class AccountRestrictedError(CreditsServiceError):
    def __init__(self, reason: str | None = None, outstanding_micro: int = 0):
        if reason == "llm_overage":
            amount = format_credits(max(int(outstanding_micro), 0))
            message = (
                "This account has an unpaid model-usage overage of "
                f"{amount} Credits. Add Credits to restore access."
            )
        else:
            message = (
                "This account is paused for a payment refund review; "
                "an administrator must restore access."
            )
        super().__init__(
            "credit_account_restricted",
            message,
        )


def _safe_log_value(value: object, *, limit: int = 120) -> str:
    """One-line, length-capped rendering for a value that came off the wire."""
    text = str(value or "")
    return text.replace("\r", " ").replace("\n", " ")[:limit]


def _log_billing(message: str) -> None:
    """Emit an operator-visible billing line.

    ``print`` rather than ``logging``: under the deployed uvicorn config in this
    repo ``logger.info`` emits nothing, so a logger call here would reproduce
    the very silence this exists to remove.
    """
    print(f"[credits] {message}", flush=True)


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
        # app.py loads the project .env after importing api_router. Resolve the
        # default gateway lazily so the first billing request sees that loaded
        # environment instead of permanently caching an import-time empty config.
        self.gateway = gateway

    def _gateway(self):
        if self.gateway is None:
            self.gateway = StripeTestGateway(load_billing_config())
        return self.gateway

    def get_balance(self, user_id: int) -> BalanceResult:
        account = self.store.ensure_account(user_id)
        projection = self._projection_model(self.store.get_balance_projection(user_id))
        state_reader = getattr(self.store, "get_account_billing_state", None)
        state = state_reader(user_id) if callable(state_reader) else account
        gateway = self.gateway
        if gateway is None:
            config = load_billing_config()
        else:
            config = getattr(gateway, "config", None)
        return BalanceResult(
            balance_micro=projection.total_available_micro,
            display_credits=projection.display_total_credits,
            **projection.model_dump(),
            account_status=state.get("account_status", account["status"]),
            billing_available=(True if config is None else bool(config.ready)),
            restriction_reason=state.get("restriction_reason"),
            outstanding_credits_micro=int(
                state.get("outstanding_credits_micro", 0) or 0
            ),
        )

    def grant_default_signup_credits(self, user_id: int) -> bool:
        """Give one account this campaign's welcome grant exactly once."""
        user_id = int(user_id)
        parts = {
            "campaign_key": DEFAULT_SIGNUP_CREDIT_CAMPAIGN,
            "user_id": user_id,
            "amount_micro": DEFAULT_SIGNUP_CREDITS_MICRO,
            "source": DEFAULT_SIGNUP_CREDIT_SOURCE,
            "reason": DEFAULT_SIGNUP_CREDIT_REASON,
        }
        result = self.store.grant_promotion_credits(
            **parts,
            operation_id=_operation_id(
                "promotion", DEFAULT_SIGNUP_CREDIT_CAMPAIGN, user_id
            ),
            idempotency_key=(
                f"promotion:{DEFAULT_SIGNUP_CREDIT_CAMPAIGN}:user:{user_id}"
            ),
            request_digest=_canonical_digest(parts),
        )
        return bool(result["created"])

    def backfill_default_signup_credits(self) -> dict[str, int]:
        """Replay the welcome campaign across every durable user account."""
        user_ids = self.store.list_user_ids()
        report = {"total": len(user_ids), "granted": 0, "existing": 0, "failed": 0}
        for user_id in user_ids:
            try:
                created = self.grant_default_signup_credits(user_id)
            except Exception:  # noqa: BLE001 - continue so one row cannot block all
                report["failed"] += 1
            else:
                report["granted" if created else "existing"] += 1
        return report

    @staticmethod
    def _month_start_iso() -> str:
        now = datetime.now(timezone.utc)
        return now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()

    @staticmethod
    def _projection_model(value: Mapping[str, Any]) -> BalanceProjection:
        payload = dict(value)
        payload.setdefault(
            "display_grant_credits", format_credits(payload["grant_available_micro"])
        )
        payload.setdefault(
            "display_purchased_credits",
            format_credits(payload["purchased_available_micro"]),
        )
        payload.setdefault(
            "display_total_credits", format_credits(payload["total_available_micro"])
        )
        return BalanceProjection(**payload)

    @staticmethod
    def _llm_reservation_model(value: Mapping[str, Any]) -> LLMReservation:
        return LLMReservation(
            reservation_id=str(value["reservation_id"]),
            user_id=int(value["user_id"]),
            run_id=str(value["run_id"]),
            call_index=int(value["call_index"]),
            provider_id=(
                str(value["provider_id"])
                if value.get("provider_id") is not None
                else None
            ),
            attempt_index=int(value.get("attempt_index") or 0),
            reserved_micro=int(value["reserved_micro"]),
            settled_micro=int(value["settled_micro"]),
            actual_micro=int(value.get("actual_micro") or 0),
            outstanding_micro=int(value.get("outstanding_micro") or 0),
            outstanding_recovered_micro=int(
                value.get("outstanding_recovered_micro") or 0
            ),
            status=str(value["status"]),
            created_at=str(value["created_at"]),
            updated_at=str(value["updated_at"]),
            failure_reason=value.get("failure_reason"),
        )

    @staticmethod
    def _llm_settlement_model(value: Mapping[str, Any]) -> LLMSettlementResult:
        return LLMSettlementResult(
            reservation_id=str(value["reservation_id"]),
            user_id=int(value["user_id"]),
            run_id=str(value["run_id"]),
            provider_id=(
                str(value["provider_id"])
                if value.get("provider_id") is not None
                else None
            ),
            attempt_index=int(value.get("attempt_index") or 0),
            reserved_micro=int(value["reserved_micro"]),
            settled_micro=int(value["settled_micro"]),
            actual_micro=int(value.get("actual_micro") or 0),
            outstanding_micro=int(value.get("outstanding_micro") or 0),
            outstanding_recovered_micro=int(
                value.get("outstanding_recovered_micro") or 0
            ),
            released_micro=int(value["released_micro"]),
            status=str(value["status"]),
            grant_debited_micro=int(value.get("grant_debited_micro") or 0),
            purchased_debited_micro=int(
                value.get("purchased_debited_micro") or 0
            ),
            ledger_entry_ids=tuple(
                int(item) for item in value.get("ledger_entry_ids", ())
            ),
        )

    @staticmethod
    def _pool_summary_model(value: Mapping[str, Any]) -> GrantPoolSummary:
        payload = dict(value)
        payload.setdefault(
            "display_pool_available_credits",
            format_credits(payload["pool_available_micro"]),
        )
        payload.setdefault(
            "display_allocated_to_users_credits",
            format_credits(payload["allocated_to_users_micro"]),
        )
        payload.setdefault(
            "display_assigned_this_month_credits",
            format_credits(payload["assigned_this_month_micro"]),
        )
        payload.setdefault(
            "display_reclaimed_this_month_credits",
            format_credits(payload["reclaimed_this_month_micro"]),
        )
        return GrantPoolSummary(**payload)

    def get_balance_projections(self, user_ids: list[int]) -> dict[int, BalanceProjection]:
        projections = self.store.get_balance_projections(user_ids)
        return {
            int(user_id): self._projection_model(projection)
            for user_id, projection in projections.items()
        }

    def reserve_llm_credits(
        self,
        *,
        user_id: int,
        run_id: str,
        call_index: int,
        amount_micro: int,
        provider_id: str,
        attempt_index: int = 0,
        operation_key: str | None = None,
        request_digest: str | None = None,
    ) -> LLMReservation:
        """Temporarily hold a usage ceiling; no Credit debit occurs here."""

        provider_id = validate_provider_id(provider_id)
        if (
            isinstance(attempt_index, bool)
            or not isinstance(attempt_index, int)
            or attempt_index < 0
        ):
            raise ValueError("attempt_index must be a non-negative integer")
        operation_key = operation_key or _operation_id(
            "llm_reserve", user_id, run_id, call_index, attempt_index, provider_id
        )
        request_digest = request_digest or _canonical_digest(
            {
                "user_id": int(user_id),
                "run_id": run_id,
                "call_index": call_index,
                "attempt_index": attempt_index,
                "provider_id": provider_id,
                "amount_micro": amount_micro,
            }
        )
        reservation_id = _operation_id(
            "llm_res",
            user_id,
            run_id,
            call_index,
            attempt_index,
            provider_id,
            operation_key,
        )
        raw = self.store.reserve_llm_credits(
            reservation_id=reservation_id,
            user_id=int(user_id),
            run_id=str(run_id),
            call_index=int(call_index),
            attempt_index=int(attempt_index),
            provider_id=provider_id,
            reserved_micro=int(amount_micro),
            operation_key=operation_key,
            request_digest=request_digest,
        )
        result = self._llm_reservation_model(raw)
        self._emit_credit_buckets(
            event_name="credits_reserved",
            raw=raw,
            amounts={
                "grant": int(raw.get("reserved_grant_micro") or 0),
                "purchased": int(raw.get("reserved_purchased_micro") or 0),
            },
        )
        return result

    def settle_llm_credits(
        self,
        reservation_id: str,
        *,
        actual_micro: int,
        evidence: Mapping[str, Any],
    ) -> LLMSettlementResult:
        """Debit the held amount, recording and restricting on any overage."""

        raw = self.store.settle_llm_credits(
            reservation_id,
            actual_micro=int(actual_micro),
            evidence=dict(evidence),
        )
        result = self._llm_settlement_model(raw)
        self._emit_credit_buckets(
            event_name="credits_settled",
            raw=raw,
            amounts={
                "grant": result.grant_debited_micro,
                "purchased": result.purchased_debited_micro,
            },
        )
        self._emit_released_credit_buckets(raw, result)
        return result

    def release_llm_credits(
        self,
        reservation_id: str,
        *,
        reason: str,
    ) -> LLMSettlementResult:
        """Release an unused usage ceiling without creating a debit."""

        raw = self.store.release_llm_credits(reservation_id, reason=reason)
        result = self._llm_settlement_model(raw)
        self._emit_released_credit_buckets(raw, result)
        return result

    def release_run_llm_reservations(
        self,
        run_id: str,
        *,
        reason: str,
    ) -> list[LLMSettlementResult]:
        results = []
        for raw in self.store.release_run_llm_reservations(run_id, reason=reason):
            result = self._llm_settlement_model(raw)
            self._emit_released_credit_buckets(raw, result)
            results.append(result)
        return results

    @staticmethod
    def _emit_credit_buckets(
        *,
        event_name: str,
        raw: Mapping[str, Any],
        amounts: Mapping[str, int],
    ) -> None:
        for bucket in ("grant", "purchased"):
            amount_micro = int(amounts.get(bucket) or 0)
            if amount_micro <= 0:
                continue
            analytics_instrumentation.emit_resource_event(
                event_name=event_name,
                user_id=int(raw["user_id"]),
                source_record_type="credit_reservation",
                source_record_id=str(raw["reservation_id"]),
                correlation_id=str(raw["run_id"]),
                billing_mode="platform_credits",
                properties={
                    "amount_micro": amount_micro,
                    "bucket": bucket,
                },
                version=bucket,
            )

    @classmethod
    def _emit_released_credit_buckets(
        cls,
        raw: Mapping[str, Any],
        result: LLMSettlementResult,
    ) -> None:
        cls._emit_credit_buckets(
            event_name="credits_refunded",
            raw=raw,
            amounts={
                "grant": max(
                    0,
                    int(raw.get("reserved_grant_micro") or 0)
                    - result.grant_debited_micro,
                ),
                "purchased": max(
                    0,
                    int(raw.get("reserved_purchased_micro") or 0)
                    - result.purchased_debited_micro,
                ),
            },
        )

    def get_grant_pool_summary(
        self, pool_id: str = "default", month_start_iso: str | None = None
    ) -> GrantPoolSummary:
        boundary = month_start_iso or self._month_start_iso()
        return self._pool_summary_model(
            self.store.get_grant_pool_summary(pool_id, boundary)
        )

    def list_grant_pool_activity(
        self, pool_id: str = "default", *, limit: int = 50, cursor: int | None = None
    ) -> dict[str, Any]:
        return self.store.list_grant_pool_activity(
            pool_id, limit=limit, cursor=cursor
        )

    def _grant_command(
        self,
        *,
        operation: str,
        admin_id: int,
        user_id: int | None,
        request: Any,
    ) -> dict[str, Any]:
        parts = {
            "operation": operation,
            "actor_user_id": int(admin_id),
            "pool_id": getattr(request, "pool_id", "default"),
            "user_id": user_id,
            "amount_micro": request.amount_micro,
            "source": request.source,
            "reason": request.reason,
        }
        return {
            **{key: value for key, value in parts.items() if key != "operation"},
            "operation_id": _operation_id(
                f"grant_{operation}", request.client_request_id
            ),
            "idempotency_key": f"admin-grant:{request.client_request_id}",
            "request_digest": _canonical_digest(parts),
        }

    def _grant_result(
        self, *, operation: str, raw: Mapping[str, Any]
    ) -> GrantMutationResult:
        entry = raw["entry"]
        operation_types = {
            "fund": "fund_grant_pool",
            "reduce": "reduce_grant_pool",
            "assign": "assign_grant",
            "reclaim": "reclaim_grant",
        }
        summary = None
        if raw.get("pool") is not None:
            summary = self.get_grant_pool_summary(str(entry["pool_id"]))
        user_balance = raw.get("user_balance")
        return GrantMutationResult(
            operation_id=entry["operation_id"],
            operation_type=operation_types[operation],
            actor_user_id=int(entry["actor_user_id"]),
            target_user_id=(
                int(entry["user_id"]) if entry.get("user_id") is not None else None
            ),
            amount_micro=abs(int(entry["amount_micro"])),
            source=entry["source"],
            reason=entry["reason"],
            created_at=entry["created_at"],
            pool=summary,
            user_balance=(
                self._projection_model(user_balance) if user_balance is not None else None
            ),
            pool_ledger_entry_id=int(entry["id"]),
            user_ledger_entry_id=(
                int(entry["user_ledger_entry_id"])
                if entry.get("user_ledger_entry_id") is not None
                else None
            ),
            recovery=(
                CreditRecoveryResult.model_validate(raw["recovery"])
                if raw.get("recovery") is not None
                else None
            ),
        )

    def fund_grant_pool(self, *, admin_id: int, request: Any) -> GrantMutationResult:
        command = self._grant_command(
            operation="fund", admin_id=admin_id, user_id=None, request=request
        )
        return self._grant_result(
            operation="fund", raw=self.store.fund_grant_pool(**command)
        )

    def reduce_grant_pool(self, *, admin_id: int, request: Any) -> GrantMutationResult:
        command = self._grant_command(
            operation="reduce", admin_id=admin_id, user_id=None, request=request
        )
        return self._grant_result(
            operation="reduce", raw=self.store.reduce_grant_pool(**command)
        )

    def assign_grant(
        self, *, admin_id: int, user_id: int, request: Any
    ) -> GrantMutationResult:
        command = self._grant_command(
            operation="assign", admin_id=admin_id, user_id=user_id, request=request
        )
        return self._grant_result(
            operation="assign", raw=self.store.assign_grant(**command)
        )

    def reclaim_grant(
        self, *, admin_id: int, user_id: int, request: Any
    ) -> GrantMutationResult:
        command = self._grant_command(
            operation="reclaim", admin_id=admin_id, user_id=user_id, request=request
        )
        return self._grant_result(
            operation="reclaim",
            raw=self.store.reclaim_grant(**command),
        )

    def list_ledger(
        self,
        user_id: int,
        *,
        limit: int,
        cursor: str | int | None,
    ):
        return self.store.list_ledger_entries(user_id, limit=limit, cursor=cursor)

    def get_order(self, order_id: str, user_id: int):
        return self.store.get_order_for_user(order_id, user_id)

    def list_admin_orders(self, *, limit: int, cursor: int | None):
        return self.store.list_orders_for_admin(limit=limit, cursor=cursor)

    def reinstate_account(self, user_id: int) -> BalanceResult:
        """Admin remedy for an automatic restriction; returns the fresh balance."""
        state_reader = getattr(self.store, "get_account_billing_state", None)
        if callable(state_reader):
            state = state_reader(user_id)
            if state.get("restriction_reason") == "llm_overage":
                raise CreditsServiceError(
                    "credit_account_requires_recovery",
                    "Model-usage overage must be settled with added Credits before reinstatement.",
                )
        self.store.reinstate_account(user_id)
        return self.get_balance(user_id)

    def create_checkout(self, user_id: int, request: CheckoutRequest) -> CheckoutResult:
        # The restriction has to be enforced here, not only in the browser.
        # credits.js disables the purchase buttons on a restricted account, but
        # that is a hint, not a gate: an account restricted by
        # _reconcile_external_refund (a refund larger than the refundable lot)
        # could still create Checkout Sessions with a plain fetch.
        account = self.store.ensure_account(user_id)
        if account["status"] == "restricted":
            state_reader = getattr(self.store, "get_account_billing_state", None)
            state = state_reader(user_id) if callable(state_reader) else account
            if state.get("restriction_reason") != "llm_overage":
                raise AccountRestrictedError(
                    state.get("restriction_reason"),
                    int(state.get("outstanding_credits_micro", 0) or 0),
                )

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

        session = self._gateway().create_checkout_session(
            order_id=order["id"],
            user_reference=str(user_id),
            amount_usd_cents=order["amount_usd_cents"],
            credits_micro=order["credits_micro"],
            idempotency_key=f"checkout:{order['id']}",
        )
        # Stripe has created a live, payable Checkout Session by this line, so
        # a failure recording its id must not fail the request: that would hand
        # the caller a 5xx while a session they may already hold stays payable.
        # settle_paid_checkout claims a NULL session id for exactly this window,
        # so the purchase still settles from the signed webhook. A genuine
        # OrderConflictError (a *different* session already attached) is a real
        # conflict and still propagates.
        try:
            updated = self.store.attach_checkout_session(
                order["id"], checkout_session_id=session.session_id
            )
        except OrderConflictError:
            raise
        except Exception as exc:
            _log_billing(
                "ERROR could not record Checkout Session for order "
                f"{_safe_log_value(order['id'])}: {type(exc).__name__}; "
                "the session stays payable and settles from the webhook"
            )
            updated = order
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
        # A reservation is subtracted from the order's refundable lot the moment
        # it is written (only pending/submitted/succeeded rows count), so a
        # reservation nobody ever clears understates the lot forever.
        #
        # Releasing it is only safe when Stripe definitively refused the
        # request. For an ambiguous failure — a timeout, a dropped connection —
        # a Refund may exist despite the error, and releasing would let the same
        # money be refunded twice. Those are deliberately *kept*: refund_id is
        # derived from the caller's client_request_id, so retrying with the same
        # client_request_id reuses this reservation and Stripe's own idempotency
        # key, which is the designed recovery path.
        try:
            result = self._gateway().create_refund(
                refund_id=reservation["id"],
                payment_intent_id=payment_intent_id,
                amount_usd_cents=reservation["amount_usd_cents"],
                idempotency_key=f"refund:{reservation['id']}",
            )
        except StripeGatewayDefinitiveError:
            self._release_reservation(reservation["id"])
            raise
        if (
            result.payment_intent_id != payment_intent_id
            or result.amount_usd_cents != reservation["amount_usd_cents"]
        ):
            # The Stripe Refund does exist here, it just does not match what was
            # asked for, so the reservation is NOT released: the money may be in
            # flight and freeing the lot could let an admin over-refund.
            _log_billing(
                "ERROR Stripe returned a mismatched Refund for reservation "
                f"{_safe_log_value(reservation['id'])}; reservation held for "
                "manual reconciliation"
            )
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

    def _release_reservation(self, refund_id: str) -> None:
        """Best-effort release; never mask the original gateway failure."""
        try:
            self.store.cancel_refund_reservation(refund_id)
        except Exception as exc:  # pragma: no cover - defensive
            _log_billing(
                "ERROR could not release refund reservation "
                f"{_safe_log_value(refund_id)}: {type(exc).__name__}; the "
                "order's refundable amount is understated until it is cleared"
            )

    def handle_webhook(self, payload: bytes, signature_header: str) -> WebhookResult:
        event = self._gateway().verify_webhook(payload, signature_header)
        if event.livemode:
            return self._record_event(
                event,
                outcome="rejected",
                reason="Stripe Live Mode events are not accepted",
            )
        # async_payment_succeeded must be handled wherever completed is. For a
        # delayed-notification payment method, `completed` arrives with
        # payment_status='unpaid' (recorded "Checkout is not paid") and the
        # settlement arrives later as this event; without it the customer is
        # charged and never credited. Its failure twin below was already
        # handled, so the guard against losing Credits existed and the guard
        # against losing *money* did not. Unreachable while
        # payment_method_types is ['card'] — which is exactly why it has to be
        # here before someone adds a method and makes it live silently.
        if event.event_type in {
            "checkout.session.completed",
            "checkout.session.async_payment_succeeded",
        }:
            return self._handle_checkout_completed(event)
        if event.event_type in {
            "checkout.session.expired",
            "checkout.session.async_payment_failed",
        }:
            return self._handle_checkout_unpaid(event)
        if event.event_type in {"refund.created", "refund.updated", "refund.failed"}:
            return self._handle_refund_event(event)
        return self._record_event(
            event,
            outcome="ignored",
            reason="Unsupported Stripe event type",
        )

    @staticmethod
    def _log_outcome(event: StripeWebhookEvent, outcome: str, reason: Any) -> None:
        """Make a non-settling webhook outcome visible to an operator.

        Every rejection path here answers the caller with HTTP 200
        ``{"received": true}`` — it has to, or Stripe would retry an event that
        will never be accepted. The consequence is that Stripe's dashboard shows
        "delivered" for an event ATL threw away, and the only other record is a
        ``stripe_webhook_events`` row nobody queries. So an upstream field
        rename (``amount_total``, ``payment_status``, ``metadata.atl_order_id``,
        ``payment_intent``) rejects *every* payment while customers are charged,
        with a green test suite and no error anywhere. Nothing distinguished
        "no events yet" from "every event silently rejected"; this line does.

        Deliberately unconditional rather than sampled or deduplicated: the
        failure this exists to catch is wholesale, so the signal must be
        present on the first occurrence, not after a threshold.
        """
        if outcome in {"processed", "duplicate"}:
            return
        severity = "ERROR" if outcome == "rejected" else "WARN"
        print(
            f"[credits] {severity} webhook {outcome}: "
            f"type={_safe_log_value(event.event_type)} "
            f"event={_safe_log_value(event.event_id)} "
            f"reason={_safe_log_value(reason)}",
            flush=True,
        )

    def _record_event(
        self,
        event: StripeWebhookEvent,
        *,
        outcome: str,
        reason: str,
        account_restricted: bool = False,
    ) -> WebhookResult:
        self._log_outcome(event, outcome, reason)
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
        # The store rejects on its own criteria (currency, amount, session id),
        # so its outcome needs the same visibility as the checks above.
        self._log_outcome(event, result["outcome"], result.get("reason"))
        return WebhookResult(
            outcome=result["outcome"],
            event_type=event.event_type,
            reason=result.get("reason"),
            balance_micro=result.get("balance_micro"),
            recovered_micro=int(result.get("recovered_micro", 0) or 0),
            outstanding_micro=int(result.get("outstanding_micro", 0) or 0),
        )

    def _handle_checkout_unpaid(self, event: StripeWebhookEvent) -> WebhookResult:
        obj = event.data_object
        metadata = obj.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        order_id = _required_text(metadata.get("atl_order_id"))
        client_reference_id = _required_text(obj.get("client_reference_id"))
        if event.livemode:
            return self._record_event(
                event, outcome="rejected", reason="Live Mode payment is not accepted"
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
        if _required_text(metadata.get("atl_user_reference")) != str(order["user_id"]):
            return self._record_event(
                event, outcome="rejected", reason="Checkout user metadata is invalid"
            )
        terminal_status = (
            "expired"
            if event.event_type == "checkout.session.expired"
            else "failed"
        )
        result = self.store.settle_unpaid_checkout(
            event_id=event.event_id,
            event_type=event.event_type,
            livemode=event.livemode,
            object_id=event.object_id,
            payload_sha256=event.payload_sha256,
            order_id=order_id,
            checkout_session_id=event.object_id,
            terminal_status=terminal_status,
        )
        self._log_outcome(event, result["outcome"], result.get("reason"))
        return WebhookResult(
            outcome=result["outcome"],
            event_type=event.event_type,
            reason=result.get("reason"),
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
        self._log_outcome(event, result["outcome"], result.get("reason"))
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
