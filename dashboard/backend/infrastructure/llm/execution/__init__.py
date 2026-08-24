"""Typed, provider-neutral execution contracts for real model calls."""

from .errors import ExecutionErrorCategory, LLMExecutionError
from .client import AnthropicCompatibleExecutionClient
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
    LLMMessage,
    LLMUsage,
    PricingSnapshot,
    UsagePolicy,
)

__all__ = [
    "BillingEvidence",
    "AnthropicCompatibleExecutionClient",
    "BillingMode",
    "ExecutionErrorCategory",
    "ExecutionHandoff",
    "ExecutionHandoffError",
    "HandoffReplayGuard",
    "LLMExecutionError",
    "LLMExecutionRequest",
    "LLMExecutionResult",
    "LLMMessage",
    "LLMUsage",
    "PricingSnapshot",
    "UsagePolicy",
    "consume_execution_handoff",
    "create_execution_handoff",
]
