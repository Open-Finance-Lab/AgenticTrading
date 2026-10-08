"""Safe, fixed error categories for model execution."""

from __future__ import annotations

from enum import StrEnum

# ``RetryHint`` lives in the leaf ``http_policy``; re-exported here.
from dashboard.backend.infrastructure.llm.http_policy import RetryHint


class ExecutionErrorCategory(StrEnum):
    CREDENTIAL_MISSING = "credential_missing"
    CREDENTIAL_INVALID = "credential_invalid"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    RESPONSE_INVALID = "response_invalid"
    USAGE_UNAVAILABLE = "usage_unavailable"
    BILLING_FAILED = "billing_failed"
    PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
    ACCOUNT_RESTRICTED = "account_restricted"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    WORKER_FAILED = "worker_failed"



_SAFE_MESSAGES = {
    ExecutionErrorCategory.CREDENTIAL_MISSING: "The selected model credential is unavailable.",
    ExecutionErrorCategory.CREDENTIAL_INVALID: "The selected model credential is invalid.",
    ExecutionErrorCategory.PROVIDER_UNAVAILABLE: "The selected model provider is unavailable.",
    ExecutionErrorCategory.PROVIDER_TIMEOUT: "The selected model provider timed out.",
    ExecutionErrorCategory.RESPONSE_INVALID: "The model returned an invalid response.",
    ExecutionErrorCategory.USAGE_UNAVAILABLE: "The model did not return billable usage.",
    ExecutionErrorCategory.BILLING_FAILED: "Model usage billing could not be completed.",
    ExecutionErrorCategory.PROVIDER_QUOTA_EXHAUSTED: (
        "The selected model provider has insufficient balance or quota."
    ),
    ExecutionErrorCategory.ACCOUNT_RESTRICTED: (
        "Your Credits account is paused. Add Credits to settle model usage "
        "or contact an administrator."
    ),
    # The balance cannot cover the next call's reservation. Kept apart from
    # BILLING_FAILED, which it used to fall into via the catch-all in
    # ``_execute_platform``: "you are out of Credits" is the user's to fix,
    # "billing broke" is ours, and one message for both sent users hunting
    # for a bug that was an empty balance.
    ExecutionErrorCategory.INSUFFICIENT_CREDITS: (
        "Not enough ATL Credits to continue this run. Add Credits on the "
        "Credits page, choose a lower-cost model, or use your own API key."
    ),
    ExecutionErrorCategory.WORKER_FAILED: "The model worker failed before completion.",
}

_SAFE_RESTRICTED_MESSAGES = {
    "llm_overage": "Your Credits account is paused because model usage exceeded its reserved amount. Add Credits to settle the outstanding usage.",
    "refund_reconciliation": (
        "Your Credits account is paused for payment refund review. "
        "Contact an administrator to restore access."
    ),
}


class LLMExecutionError(RuntimeError):
    """An expected execution failure whose message never contains upstream data."""

    def __init__(
        self,
        category: ExecutionErrorCategory | str,
        message: str | None = None,
    ) -> None:
        self.category = ExecutionErrorCategory(category)
        allowed_message = (
            message
            if message in (*_SAFE_MESSAGES.values(), *_SAFE_RESTRICTED_MESSAGES.values())
            else None
        )
        if (
            category == ExecutionErrorCategory.ACCOUNT_RESTRICTED
            and isinstance(message, str)
            and message.startswith(
                "Your Credits account is paused because model usage exceeded its reserved amount."
            )
        ):
            allowed_message = message
        self.safe_message = allowed_message or _SAFE_MESSAGES[self.category]
        # Set by ``account_restricted``; lets a process boundary that can carry
        # only fixed tokens (the backtest child's run_failed line) rebuild the
        # restriction-specific message rather than the generic one.
        self.restriction_reason: str | None = None
        super().__init__(self.safe_message)

    @classmethod
    def safe(cls, category: ExecutionErrorCategory | str) -> "LLMExecutionError":
        return cls(category)

    @classmethod
    def account_restricted(
        cls, reason: str | None, outstanding_micro: int = 0
    ) -> "LLMExecutionError":
        if reason == "llm_overage" and outstanding_micro > 0:
            whole, fraction = divmod(int(outstanding_micro), 1_000_000)
            message = (
                "Your Credits account is paused because model usage exceeded its "
                f"reserved amount. Add at least {whole}.{fraction:06d} Credits "
                "to settle the outstanding usage."
            )
        else:
            message = _SAFE_RESTRICTED_MESSAGES.get(
                reason, _SAFE_MESSAGES[ExecutionErrorCategory.ACCOUNT_RESTRICTED]
            )
        error = cls(ExecutionErrorCategory.ACCOUNT_RESTRICTED, message)
        if reason in _SAFE_RESTRICTED_MESSAGES:
            error.restriction_reason = reason
        return error


# A hosted (``fail_closed``) run aborts on any model error, because a billing or
# credential failure absorbed as a step would let the run continue unbilled or
# unauthorised. A provider outage is neither: the failed call's Credits
# reservation is already released before the error reaches the caller, so
# absorbing it costs one held step and nothing else.
_TRANSIENT_PROVIDER_CATEGORIES = frozenset(
    {
        ExecutionErrorCategory.PROVIDER_UNAVAILABLE,
        ExecutionErrorCategory.PROVIDER_TIMEOUT,
    }
)


def is_transient_provider_failure(error: BaseException) -> bool:
    """True for a provider outage a hosted run may hold through."""
    return (
        isinstance(error, LLMExecutionError)
        and error.category in _TRANSIENT_PROVIDER_CATEGORIES
    )


def run_failed_line(error: LLMExecutionError) -> str:
    """The line a backtest child prints when a model call ends its run.

    Parsed by ``api/routers/backtests._child_llm_failure``. Fixed tokens only
    (enum values), never upstream text, and the ``ERROR: llm.`` prefix makes
    the parent relay it to the service log live.
    """
    line = f"ERROR: llm.run_failed category={error.category.value}"
    reason = getattr(error, "restriction_reason", None)
    if reason:
        line += f" reason={reason}"
    return line


__all__ = [
    "ExecutionErrorCategory",
    "LLMExecutionError",
    "RetryHint",
    "is_transient_provider_failure",
    "run_failed_line",
]
