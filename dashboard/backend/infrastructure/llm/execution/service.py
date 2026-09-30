"""Provider-neutral model execution with safe credential and billing lanes."""

from __future__ import annotations

import itertools
import json
import re
import threading
import time
from collections.abc import Callable, Iterator

from dashboard.backend.domain.credits.models import LLMSettlementResult
from dashboard.backend.domain.credits.service import CreditsService
from dashboard.backend.domain.credits.repository_common import (
    CreditAccountRestrictedStoreError,
)
from dashboard.backend.domain.analytics import instrumentation as analytics_instrumentation
from dashboard.backend.domain.analytics.usage_cost import model_usage_properties
from dashboard.backend.domain.model_providers.models import ProviderRecord
from dashboard.backend.domain.model_providers.service import (
    ModelProviderService,
    ResolvedCredential,
)
from dashboard.backend.infrastructure.llm.execution.adapters.base import (
    AdapterResponse,
    ProviderExecutionAdapter,
    provider_read_timeout_seconds,
)
from dashboard.backend.infrastructure.llm.execution.adapters.registry import (
    get_execution_adapter,
)
from dashboard.backend.infrastructure.llm.execution.errors import (
    ExecutionErrorCategory,
    LLMExecutionError,
    RetryHint,
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

# Same-provider retries. This service is the only retry owner: the SDKs run
# with max_retries=0 (see the provider-timeout block in adapters/base.py),
# because their retry loop replays read timeouts -- a whole billed generation
# -- and sends no idempotency key. Here an attempt is repeated at the same
# provider only when ``map_provider_error`` says nothing was generated:
#   - ``RetryHint.PRE_SEND`` (DNS, connect, TLS, a stalled body), at any age;
#   - ``RetryHint.REJECTED`` (408/409/429/5xx, a dropped connection) only when
#     it came back within FAST_FAILURE_SECONDS. CommonStack's 2026-09-27 500s
#     arrived ~24s in, after the generation, and three SDK replays of them
#     bought nothing but 73s; a gateway that refuses outright answers in <2s,
#     and the fastest DeepSeek completion seen is ~24s. The gate is calibrated
#     on DeepSeek: a model that can finish in under 15s (Haiku, GPT-5.5) can
#     have a fast post-generation 5xx or cut-off body repeated. That is still
#     fewer replays than the SDK made unconditionally, each on its own
#     reservation, and fail_closed makes a run abort the costlier outcome.
# A read timeout is never repeated here: failover to the next candidate is the
# only second chance, and it is a recorded, reserved one. Every attempt --
# repeat or failover -- takes the next ``attempt_index`` of its call, because
# a reservation is keyed on (user, run, call, attempt) and reusing an index
# returns the already-released row (BILLING_FAILED).
MAX_SAME_PROVIDER_RETRIES = 2
SAME_PROVIDER_BACKOFF_SECONDS = (4.0, 12.0)
FAST_FAILURE_SECONDS = 15.0
# A stated Retry-After is waited out up to the SDKs' own ceiling (openai and
# anthropic ``_calculate_retry_timeout`` honour <= 60s), so no refusal the SDK
# used to wait out now fails the call -- which matters because "fail over
# instead" means "abort the run" on BYOK and whenever the next lane is dead
# (OpenRouter, #523). Above it the provider is saying "not this minute", and
# repeating early would only be refused again.
RETRY_AFTER_CAP_SECONDS = 60.0
# Failures of this one call at this one provider, as opposed to lane state
# (quota, credentials) that is equally true of every call.
_CALL_SPECIFIC_FAILURES = frozenset(
    {
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
    }
)
_LOG_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _report_provider_attempt_failed(
    request: LLMExecutionRequest,
    attempt_index: int,
    exc: LLMExecutionError,
    next_step: str,
) -> None:
    """Print one line per failed attempt; the parent relays ``ERROR: llm.`` lines live.

    Not deduplicated: each line is a distinct failed attempt, and the count is
    bounded by the retry policy above. ``elapsed_s=-`` marks one that failed
    before reaching the provider (provider/adapter resolution): it holds an
    ``attempt_index`` but no reservation row. Every field is a
    validated identifier, an enum value or a number -- never exception text,
    and never ``=``/``:`` inside a value, so ``_redact_credentials`` in the
    parent has nothing to rewrite.
    """

    elapsed = getattr(exc, "provider_elapsed_seconds", None)
    status = getattr(exc, "provider_status_code", None)
    hint = getattr(exc, "retry_hint", RetryHint.NONE)
    run_id = request.run_id if _LOG_SAFE_RUN_ID.match(request.run_id) else "-"
    print(
        "ERROR: llm.provider_attempt_failed "
        f"run={run_id} call={request.call_index} attempt={attempt_index} "
        f"provider={request.provider_id} model={request.model_id} "
        f"billing={request.billing_mode.value} category={exc.category.value} "
        f"phase={getattr(exc, 'timeout_phase', None) or '-'} "
        f"status={status if isinstance(status, int) else '-'} "
        f"hint={RetryHint(hint).value} "
        f"elapsed_s={f'{elapsed:.1f}' if isinstance(elapsed, float) else '-'} "
        f"read_timeout_s={provider_read_timeout_seconds()} next={next_step}",
        flush=True,
    )


# CommonStack has no balance endpoint (every candidate path 404s as of
# 2026-09-23), so a drained platform lane is only observable as a failed
# call. Once per provider per process: a backtest child is one run, so a
# drained lane costs one line per run, not one per call.
_quota_exhausted_reported: set[str] = set()
_quota_exhausted_lock = threading.Lock()


def _report_platform_quota_exhausted(provider_id: str, fallback: str | None) -> None:
    with _quota_exhausted_lock:
        if provider_id in _quota_exhausted_reported:
            return
        _quota_exhausted_reported.add(provider_id)
    print(
        "ERROR: llm.platform_quota_exhausted "
        f"provider={provider_id} fallback={fallback or 'none'}",
        flush=True,
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
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = providers
        self.credits = credits
        self.adapter_resolver = adapter_resolver
        self.pricing_snapshot_factory = pricing_snapshot_factory
        self._sleep = sleep
        self._clock = clock
        self._platform_runs: set[str] = set()

    def execute(self, request: LLMExecutionRequest) -> LLMExecutionResult:
        """Run one logical model call with its selected payment lane.

        One logical call is up to 1 + MAX_SAME_PROVIDER_RETRIES physical
        attempts per candidate, repeated only for pre-send failures and fast
        rejections, each billed on its own reservation; Platform Credits then
        fail over across the route's candidates.
        """
        try:
            if request.billing_mode is BillingMode.PLATFORM_CREDITS:
                result = self._execute_with_platform_failover(request)
            else:
                result = self._execute_candidate(
                    request,
                    attempts=itertools.count(),
                    requested_provider_id=request.provider_id,
                    next_provider_id=None,
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
        # BYOK debits no Credits, so its cost_micro_usd is 0; its list-price
        # estimate travels in a separate property (analytics.usage_cost). The
        # provider cost of a BYOK call belongs to the user's own key and is not
        # the platform's equivalent, so it is deliberately not recorded.
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
            properties=model_usage_properties(
                billing_mode=request.billing_mode.value,
                model_id=request.model_id,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                usage_available=result.usage.usage_available,
                provider_cost_usd=result.billing.provider_cost_usd,
                estimated_cost_usd=result.billing.estimated_cost_usd,
            ),
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

    @staticmethod
    def _same_provider_retry_delay(
        exc: LLMExecutionError,
        retries_done: int,
    ) -> float | None:
        """Seconds to wait before repeating ``exc``'s attempt, or None to stop."""

        if retries_done >= MAX_SAME_PROVIDER_RETRIES:
            return None
        if exc.category not in _CALL_SPECIFIC_FAILURES:
            return None
        hint = getattr(exc, "retry_hint", RetryHint.NONE)
        elapsed = getattr(exc, "provider_elapsed_seconds", None)
        if hint is RetryHint.PRE_SEND:
            pass
        elif (
            hint is RetryHint.REJECTED
            and isinstance(elapsed, float)
            and elapsed <= FAST_FAILURE_SECONDS
        ):
            pass
        else:
            # A read timeout, a slow rejection, or anything unclassified.
            return None
        delay = SAME_PROVIDER_BACKOFF_SECONDS[retries_done]
        retry_after = getattr(exc, "retry_after_seconds", None)
        if retry_after is not None:
            if retry_after > RETRY_AFTER_CAP_SECONDS:
                return None
            delay = max(delay, retry_after)
        return delay

    def _execute_candidate(
        self,
        request: LLMExecutionRequest,
        *,
        attempts: Iterator[int],
        requested_provider_id: str,
        next_provider_id: str | None,
    ) -> LLMExecutionResult:
        """Run ``request`` at its provider, repeating only what generated nothing."""

        retries_done = 0
        while True:
            attempt_index = next(attempts)
            try:
                return self._execute_once(
                    request,
                    attempt_index=attempt_index,
                    requested_provider_id=requested_provider_id,
                )
            except LLMExecutionError as exc:
                delay = self._same_provider_retry_delay(exc, retries_done)
                if exc.category in _CALL_SPECIFIC_FAILURES:
                    _report_provider_attempt_failed(
                        request,
                        attempt_index,
                        exc,
                        "retry" if delay is not None else (next_provider_id or "none"),
                    )
                if delay is None:
                    raise
                self._sleep(delay)
                retries_done += 1

    def _execute_with_platform_failover(
        self,
        request: LLMExecutionRequest,
    ) -> LLMExecutionResult:
        """Try ordered platform candidates, retaining one requested identity."""

        # The route's ordered tuple is authoritative: it already applied
        # ATL_PLATFORM_PROVIDER_ORDER against the registry, and the handoff
        # carries it whole. The worker used to re-derive routing here, turning
        # a lone ("openrouter",) into OpenRouter -> CommonStack -- hard-coded
        # OpenRouter-first, and blind to an order the route had rejected. A
        # lone candidate is what the route decided, so it is not widened.
        candidates = tuple(request.provider_ids or (request.provider_id,))
        requested_provider_id = candidates[0]
        # One counter for the whole call: same-provider repeats and failover
        # share it (see MAX_SAME_PROVIDER_RETRIES). With no repeats it is the
        # candidate position, as it always was.
        attempts = itertools.count()
        first_call_specific: LLMExecutionError | None = None
        last_error: LLMExecutionError | None = None
        for position, provider_id in enumerate(candidates):
            next_provider_id = (
                candidates[position + 1] if position + 1 < len(candidates) else None
            )
            attempt_request = request.model_copy(
                update={
                    "provider_id": provider_id,
                    "provider_ids": candidates,
                }
            )
            try:
                return self._execute_candidate(
                    attempt_request,
                    attempts=attempts,
                    requested_provider_id=requested_provider_id,
                    next_provider_id=next_provider_id,
                )
            except LLMExecutionError as exc:
                last_error = exc
                if exc.category is ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED:
                    _report_platform_quota_exhausted(provider_id, next_provider_id)
                if exc.category not in _PLATFORM_FAILOVER_CATEGORIES:
                    raise
                if (
                    first_call_specific is None
                    and exc.category in _CALL_SPECIFIC_FAILURES
                ):
                    first_call_specific = exc
        assert last_error is not None
        # The first candidate whose final error was a timeout/unavailable
        # outranks later lane state: a quota-dead fallback must not relabel
        # the primary's timeout as "insufficient balance" (09-28: CommonStack
        # timed out, OpenRouter answered 402, and the run reported quota). The
        # quota line above still names the drained lane.
        raise first_call_specific or last_error

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

    def _complete(
        self,
        adapter: ProviderExecutionAdapter,
        request: LLMExecutionRequest,
        credential: ResolvedCredential,
        provider: ProviderRecord,
    ) -> AdapterResponse:
        # Time the provider call alone -- not credential, pricing or ledger
        # work -- because the same-provider retry gate reads it.
        started = self._clock()
        try:
            response = adapter.complete(request, credential, provider)
        except LLMExecutionError as exc:
            exc.provider_elapsed_seconds = float(self._clock() - started)
            raise
        except Exception as exc:  # noqa: BLE001 - adapter bugs cannot leak details
            error = LLMExecutionError(ExecutionErrorCategory.PROVIDER_UNAVAILABLE)
            error.provider_elapsed_seconds = float(self._clock() - started)
            raise error from exc
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
