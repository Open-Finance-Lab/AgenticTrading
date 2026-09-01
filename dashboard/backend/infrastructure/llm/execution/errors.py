"""Safe, fixed error categories for model execution."""

from __future__ import annotations

from enum import StrEnum


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
        return cls(ExecutionErrorCategory.ACCOUNT_RESTRICTED, message)


__all__ = ["ExecutionErrorCategory", "LLMExecutionError"]
