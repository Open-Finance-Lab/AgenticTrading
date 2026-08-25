"""Typed, provider-neutral execution contracts for real model calls."""

from .errors import ExecutionErrorCategory, LLMExecutionError
from .handoff import (
    ExecutionHandoff,
    ExecutionHandoffError,
    HandoffReplayGuard,
    consume_execution_handoff,
    create_execution_handoff,
)
from .models import (
    BillingEvidence,
    BillingMode,
    LLMExecutionRequest,
    LLMExecutionResult,
    LLMRunEvidence,
    LLMMessage,
    LLMUsage,
    PricingSnapshot,
    UsagePolicy,
)

__all__ = [
    "BillingEvidence",
    "BillingMode",
    "ExecutionErrorCategory",
    "ExecutionHandoff",
    "ExecutionHandoffError",
    "HandoffReplayGuard",
    "LLMExecutionError",
    "LLMExecutionRequest",
    "LLMExecutionResult",
    "LLMRunEvidence",
    "LLMMessage",
    "LLMUsage",
    "PricingSnapshot",
    "UsagePolicy",
    "consume_execution_handoff",
    "create_execution_handoff",
]
