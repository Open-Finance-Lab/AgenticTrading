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
    WORKER_FAILED = "worker_failed"


_SAFE_MESSAGES = {
    ExecutionErrorCategory.CREDENTIAL_MISSING: "The selected model credential is unavailable.",
    ExecutionErrorCategory.CREDENTIAL_INVALID: "The selected model credential is invalid.",
    ExecutionErrorCategory.PROVIDER_UNAVAILABLE: "The selected model provider is unavailable.",
    ExecutionErrorCategory.PROVIDER_TIMEOUT: "The selected model provider timed out.",
    ExecutionErrorCategory.RESPONSE_INVALID: "The model returned an invalid response.",
    ExecutionErrorCategory.USAGE_UNAVAILABLE: "The model did not return billable usage.",
    ExecutionErrorCategory.BILLING_FAILED: "Model usage billing could not be completed.",
    ExecutionErrorCategory.WORKER_FAILED: "The model worker failed before completion.",
}


class LLMExecutionError(RuntimeError):
    """An expected execution failure whose message never contains upstream data."""

    def __init__(
        self,
        category: ExecutionErrorCategory | str,
        message: str | None = None,
    ) -> None:
        self.category = ExecutionErrorCategory(category)
        allowed_message = message if message in _SAFE_MESSAGES.values() else None
        self.safe_message = allowed_message or _SAFE_MESSAGES[self.category]
        super().__init__(self.safe_message)

    @classmethod
    def safe(cls, category: ExecutionErrorCategory | str) -> "LLMExecutionError":
        return cls(category)


__all__ = ["ExecutionErrorCategory", "LLMExecutionError"]
