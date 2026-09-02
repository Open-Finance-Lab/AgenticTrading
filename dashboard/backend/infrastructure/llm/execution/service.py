"""Provider-neutral model execution with safe credential and billing lanes."""

from __future__ import annotations

import json
from collections.abc import Callable

from dashboard.backend.domain.credits.models import LLMSettlementResult
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
)
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.domain.model_providers.execution_catalog import (
    UnsupportedExecutionModel,
)
from dashboard.backend.domain.model_providers.repository_common import (
    ProviderNotFoundError,
)
from dashboard.backend.domain.model_providers.service import (
    ModelProviderService,
    ResolvedCredential,
    CredentialResolutionError,
)
from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    AdapterResponse,
    ProviderExecutionAdapter,
)
from dashboard.backend.infrastructure.llm.execution.adapters.registry import (
    get_execution_adapter,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
)
from dashboard.backend.infrastructure.llm.execution.models import (
    BillingEvidence,
    BillingMode,
    LLMExecutionRequest,
    LLMExecutionResult,
    LLMUsage,
    PricingSnapshot,
)
from dashboard.backend.infrastructure.llm.token_cost import (
    build_cost_evidence,
    credits_micro_for_usd,
    estimate_cost_from_snapshot,
)


AdapterResolver = Callable[[ProviderRecord], ProviderExecutionAdapter]
PricingSnapshotFactory = Callable[[str, str], PricingSnapshot]

_PLATFORM_FAILOVER_CATEGORIES = frozenset(
    {
        ExecutionErrorCategory.CREDENTIAL_MISSING,
        ExecutionErrorCategory.CREDENTIAL_INVALID,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED,
    }
)


# The provider receives the serialized messages, but its tokenizer is not
# necessarily the same as ATL's estimator. Reserving the UTF-8 byte count is a
# deliberately conservative token ceiling: byte-level tokenizers cannot emit
# more tokens than bytes, and the small allowance covers provider framing.
_PROMPT_FRAMING_TOKEN_ALLOWANCE = 256


def _pricing_snapshot_for(model_id: str, provider_id: str) -> PricingSnapshot:
    return PricingSnapshot.from_model(model_id, provider_id)


def _prompt_token_ceiling(request: LLMExecutionRequest) -> int:
    payload = {
        "system_message": request.system_message,
        "messages": [message.model_dump(mode="json") for message in request.messages],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(encoded) + _PROMPT_FRAMING_TOKEN_ALLOWANCE


def _reservation_ceiling_micro(
    request: LLMExecutionRequest,
    pricing_snapshot: PricingSnapshot,
) -> int:
    """Return the maximum billable cost for this one provider request."""

    ceiling_usage = LLMUsage(
        input_tokens=_prompt_token_ceiling(request),
        output_tokens=request.usage_policy.max_output_tokens,
    )
    ceiling_cost = estimate_cost_from_snapshot(pricing_snapshot, ceiling_usage)
    if ceiling_cost is None:
        raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)
    return credits_micro_for_usd(ceiling_cost)


class LLMExecutionService:
    """Resolve one credential lane, run one model call, and settle its cost."""

    def __init__(
        self,
        *,
        providers: ModelProviderService,
        credits: CreditsService,
        adapter_resolver: AdapterResolver = get_execution_adapter,
        pricing_snapshot_factory: PricingSnapshotFactory = _pricing_snapshot_for,
    ) -> None:
        self.providers = providers
        self.credits = credits
        self.adapter_resolver = adapter_resolver
        self.pricing_snapshot_factory = pricing_snapshot_factory
        self._platform_runs: set[str] = set()

    def execute(self, request: LLMExecutionRequest) -> LLMExecutionResult:
        """Run exactly one requested model call with its selected payment lane."""
        try:
            if request.billing_mode is BillingMode.PLATFORM_CREDITS:
                result = self._execute_with_platform_failover(request)
            else:
                result = self._execute_once(
                    request,
                    attempt_index=0,
                    requested_provider_id=request.provider_id,
                )
            self._emit_model_usage(request, result)
            return result
        except LLMExecutionError as exc:
            analytics_instrumentation.emit_safe_error_event(
                user_id=request.user_id,
                source_record_type="run",
                source_record_id=request.run_id,
                error_category=self._analytics_error_category(exc.category),
                correlation_id=request.run_id,
                version=f"{request.call_index}:{exc.category.value}",
            )
            raise
        except Exception:
            analytics_instrumentation.emit_safe_error_event(
                user_id=request.user_id,
                source_record_type="run",
                source_record_id=request.run_id,
                error_category="internal_error",
                correlation_id=request.run_id,
                version=f"{request.call_index}:internal_error",
            )
            raise

    @staticmethod
    def _analytics_error_category(
        category: ExecutionErrorCategory,
    ) -> str:
        return {
            ExecutionErrorCategory.CREDENTIAL_MISSING: "credential_missing",
            ExecutionErrorCategory.CREDENTIAL_INVALID: "credential_invalid",
            ExecutionErrorCategory.PROVIDER_UNAVAILABLE: "provider_unavailable",
            ExecutionErrorCategory.PROVIDER_TIMEOUT: "provider_timeout",
            ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED: "provider_quota_exhausted",
            ExecutionErrorCategory.BILLING_FAILED: "credits_unavailable",
            ExecutionErrorCategory.ACCOUNT_RESTRICTED: "account_restricted",
        }.get(category, "internal_error")

    @staticmethod
    def _emit_model_usage(
        request: LLMExecutionRequest,
        result: LLMExecutionResult,
    ) -> None:
        cost_usd = 0.0
        if request.billing_mode is BillingMode.PLATFORM_CREDITS:
            cost_usd = (
                result.billing.provider_cost_usd
                if result.billing.provider_cost_usd is not None
                else result.billing.estimated_cost_usd or 0.0
            )
        analytics_instrumentation.emit_resource_event(
            event_name="model_usage_recorded",
            user_id=request.user_id,
            source_record_type="run",
            source_record_id=request.run_id,
            correlation_id=request.run_id,
            provider_id=result.provider_id,
            model_id=request.model_id,
            billing_mode=request.billing_mode.value,
            outcome="succeeded",
            properties={
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "cost_micro_usd": max(0, round(cost_usd * 1_000_000)),
            },
            version=request.call_index,
        )

    def finalize_run(
        self,
        run_id: str,
        *,
        billing_mode: BillingMode | None = None,
    ) -> list[LLMSettlementResult]:
        """Idempotently release any open Platform Credits reservations for a run.

        A BYOK finalizer is intentionally a no-op so that lane never mutates
        the ATL Credits ledger. Worker callers should pass their known lane;
        retaining the in-process set keeps the one-argument form safe too.
        """

        if billing_mode is BillingMode.BYOK:
            return []
        try:
            released = self.credits.release_run_llm_reservations(
                run_id,
                reason=ExecutionErrorCategory.WORKER_FAILED.value,
            )
        except Exception as exc:  # noqa: BLE001 - billing errors remain sanitized
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc
        self._platform_runs.discard(run_id)
        return released

    def _execute_platform(
        self,
        *,
        request: LLMExecutionRequest,
        provider: ProviderRecord,
        credential: ResolvedCredential,
        adapter: ProviderExecutionAdapter,
        pricing_snapshot: PricingSnapshot,
        attempt_index: int,
        requested_provider_id: str,
    ) -> LLMExecutionResult:
        reservation_id: str | None = None
        try:
            reserved_micro = _reservation_ceiling_micro(request, pricing_snapshot)
            if reserved_micro > 0:
                try:
                    reservation = self.credits.reserve_llm_credits(
                        user_id=request.user_id,
                        run_id=request.run_id,
                        call_index=request.call_index,
                        attempt_index=attempt_index,
                        provider_id=request.provider_id,
                        amount_micro=reserved_micro,
                    )
                except CreditAccountRestrictedStoreError as exc:
                    try:
                        balance = self.credits.get_balance(request.user_id)
                        reason = balance.restriction_reason
                        outstanding_micro = balance.outstanding_credits_micro
                    except Exception:  # noqa: BLE001 - keep the safe fallback
                        reason = None
                        outstanding_micro = 0
                    raise LLMExecutionError.account_restricted(
                        reason, outstanding_micro
                    ) from exc
                reservation_id = reservation.reservation_id
                self._platform_runs.add(request.run_id)
                if reservation.status != "open":
                    raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)

            response = self._complete(adapter, request, credential, provider)
            usage = self._result_usage(response)
            if not usage.usage_available:
                raise LLMExecutionError(ExecutionErrorCategory.USAGE_UNAVAILABLE)
            billing = self._build_evidence(
                request=request,
                usage=usage,
                provider_cost_usd=response.provider_cost_usd,
                pricing_snapshot=pricing_snapshot,
            )
            if billing.usage_authority == "unavailable":
                raise LLMExecutionError(ExecutionErrorCategory.USAGE_UNAVAILABLE)

            if reservation_id is not None:
                actual_micro = billing.provider_cost_credits_micro
                settlement = self._settle(
                    reservation_id, billing, actual_micro=actual_micro
                )
                billing = billing.model_copy(
                    update={
                        "debited_credits_micro": settlement.settled_micro,
                        "outstanding_credits_micro": settlement.outstanding_micro,
                    }
                )
            elif billing.provider_cost_credits_micro > 0:
                # A zero-priced snapshot must not become a paid call after the
                # fact; there is no held balance from which to settle it.
                raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)

            return self._result(
                request=request,
                credential=credential,
                usage=usage,
                billing=billing,
                text=response.text,
                finish_reason=response.finish_reason,
                requested_provider_id=requested_provider_id,
            )
        except LLMExecutionError as exc:
            self._release_after_failure(reservation_id, exc.category)
            raise
        except Exception as exc:  # noqa: BLE001 - never expose store/SDK details
            self._release_after_failure(
                reservation_id, ExecutionErrorCategory.BILLING_FAILED
            )
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc

    def _execute_once(
        self,
        request: LLMExecutionRequest,
        *,
        attempt_index: int,
        requested_provider_id: str,
    ) -> LLMExecutionResult:
        """Execute one provider attempt, including its own billing lifecycle."""

        provider = self._resolve_provider(request.provider_id)
        credential = self._resolve_credential(request)
        pricing_snapshot = self.pricing_snapshot_factory(
            request.model_id, request.provider_id
        )
        self._validate_pricing_snapshot(request, pricing_snapshot)
        adapter = self._resolve_adapter(provider)

        if request.billing_mode is BillingMode.BYOK:
            response = self._complete(adapter, request, credential, provider)
            usage = self._result_usage(response)
            billing = self._build_evidence(
                request=request,
                usage=usage,
                provider_cost_usd=response.provider_cost_usd,
                pricing_snapshot=pricing_snapshot,
            )
            return self._result(
                request=request,
                credential=credential,
                usage=usage,
                billing=billing,
                text=response.text,
                finish_reason=response.finish_reason,
                requested_provider_id=requested_provider_id,
            )

        return self._execute_platform(
            request=request,
            provider=provider,
            credential=credential,
            adapter=adapter,
            pricing_snapshot=pricing_snapshot,
            attempt_index=attempt_index,
            requested_provider_id=requested_provider_id,
        )

    def _execute_with_platform_failover(
        self,
        request: LLMExecutionRequest,
    ) -> LLMExecutionResult:
        """Try ordered platform candidates, retaining one requested identity."""

        candidates = tuple(request.provider_ids or (request.provider_id,))
        # Direct service callers predating the candidate-list handoff still get
        # the established OpenRouter -> CommonStack fallback when the route is
        # available. New handoffs always carry the complete ordered tuple.
        if candidates == ("openrouter",):
            try:
                self.providers.preflight_execution_model(
                    "commonstack", request.model_id
                )
                self.providers.preflight_platform_credential("commonstack")
            except (
                ProviderNotFoundError,
                CredentialResolutionError,
                UnsupportedExecutionModel,
            ):
                pass
            else:
                candidates = ("openrouter", "commonstack")

        requested_provider_id = candidates[0]
        last_error: LLMExecutionError | None = None
        for attempt_index, provider_id in enumerate(candidates):
            attempt_request = request.model_copy(
                update={
                    "provider_id": provider_id,
                    "provider_ids": candidates,
                }
            )
            try:
                return self._execute_once(
                    attempt_request,
                    attempt_index=attempt_index,
                    requested_provider_id=requested_provider_id,
                )
            except LLMExecutionError as exc:
                last_error = exc
                if exc.category not in _PLATFORM_FAILOVER_CATEGORIES:
                    raise
        assert last_error is not None
        raise last_error

    def _resolve_provider(self, provider_id: str) -> ProviderRecord:
        store = getattr(self.providers, "store", None)
        get_provider = getattr(store, "get_provider", None)
        if not callable(get_provider):
            raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE)
        try:
            raw_provider = get_provider(provider_id)
            if not raw_provider:
                raise LLMExecutionError(ExecutionErrorCategory.CREDENTIAL_MISSING)
            return ProviderRecord.model_validate(raw_provider)
        except LLMExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - repository validation is internal
            raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE) from exc

    def _resolve_credential(self, request: LLMExecutionRequest) -> ResolvedCredential:
        try:
            if request.billing_mode is BillingMode.BYOK:
                return self.providers.resolve_user_default_credential(
                    request.user_id, request.provider_id
                )
            return self.providers.resolve_platform_credential(request.provider_id)
        except LLMExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - credential internals are secret
            raise LLMExecutionError(ExecutionErrorCategory.CREDENTIAL_MISSING) from exc

    def _resolve_adapter(self, provider: ProviderRecord) -> ProviderExecutionAdapter:
        try:
            return self.adapter_resolver(provider)
        except LLMExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapter construction is internal
            raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE) from exc

    @staticmethod
    def _validate_pricing_snapshot(
        request: LLMExecutionRequest,
        pricing_snapshot: PricingSnapshot,
    ) -> None:
        if (
            pricing_snapshot.provider_id != request.provider_id
            or pricing_snapshot.model_id != request.model_id
        ):
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)

    @staticmethod
    def _complete(
        adapter: ProviderExecutionAdapter,
        request: LLMExecutionRequest,
        credential: ResolvedCredential,
        provider: ProviderRecord,
    ) -> AdapterResponse:
        try:
            response = adapter.complete(request, credential, provider)
        except LLMExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapter bugs cannot leak details
            raise LLMExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE) from exc
        if not isinstance(response, AdapterResponse) or not response.text.strip():
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)
        if not isinstance(response.model_id, str) or not response.model_id.strip():
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)
        return response

    @staticmethod
    def _result_usage(response: AdapterResponse) -> LLMUsage:
        if response.usage is None:
            return LLMUsage(input_tokens=0, output_tokens=0, usage_available=False)
        if not isinstance(response.usage, LLMUsage):
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID)
        return response.usage

    @staticmethod
    def _build_evidence(
        *,
        request: LLMExecutionRequest,
        usage: LLMUsage,
        provider_cost_usd: float | None,
        pricing_snapshot: PricingSnapshot,
    ) -> BillingEvidence:
        try:
            return build_cost_evidence(
                billing_mode=request.billing_mode,
                provider_id=request.provider_id,
                model_id=request.model_id,
                usage=usage,
                provider_cost_usd=provider_cost_usd,
                pricing_snapshot=pricing_snapshot,
            )
        except Exception as exc:  # noqa: BLE001 - invalid cost data stays internal
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc

    def _settle(
        self,
        reservation_id: str,
        billing: BillingEvidence,
        *,
        actual_micro: int,
    ) -> LLMSettlementResult:
        try:
            settlement = self.credits.settle_llm_credits(
                reservation_id,
                actual_micro=actual_micro,
                evidence=billing.model_dump(mode="json"),
            )
        except Exception as exc:  # noqa: BLE001 - store errors stay internal
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc
        if (
            settlement.status != "settled"
            or settlement.actual_micro != actual_micro
        ):
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED)
        return settlement

    def _release_after_failure(
        self,
        reservation_id: str | None,
        category: ExecutionErrorCategory,
    ) -> None:
        if reservation_id is None:
            return
        try:
            self.credits.release_llm_credits(
                reservation_id,
                reason=category.value,
            )
        except Exception as exc:  # noqa: BLE001 - do not leave an unknown hold
            raise LLMExecutionError(ExecutionErrorCategory.BILLING_FAILED) from exc

    @staticmethod
    def _result(
        *,
        request: LLMExecutionRequest,
        credential: ResolvedCredential,
        usage: LLMUsage,
        billing: BillingEvidence,
        text: str,
        finish_reason: str | None = None,
        requested_provider_id: str | None = None,
    ) -> LLMExecutionResult:
        try:
            return LLMExecutionResult(
                text=text,
                provider_id=request.provider_id,
                requested_provider_id=requested_provider_id,
                model_id=request.model_id,
                credential_id=credential.credential_id,
                credential_key_last_four=credential.key_last_four,
                usage=usage,
                billing=billing,
                finish_reason=finish_reason,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the fixed public contract
            raise LLMExecutionError(ExecutionErrorCategory.RESPONSE_INVALID) from exc


__all__ = ["LLMExecutionService"]
