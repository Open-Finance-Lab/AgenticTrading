"""Parallel LLM gateway integrations for the trading / leaderboard pipeline.

Each integration is a sibling module exposing the same surface:

* ``INTEGRATION_ID`` — config value for ``leaderboard.json`` ``integration``
* ``DEFAULT_MODEL`` / ``default_model_name()``
* ``make_client(anthropic_cls)`` → Anthropic-compatible client or ``None``

Callers pick a gateway via ``make_llm_client(integration=...)`` (explicit) or
omit it for legacy env auto-detect (CommonStack key → CommonStack, else native
Anthropic). OpenRouter is never auto-selected — set ``integration: "openrouter"``
on the leaderboard entry (or pass the kwarg).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from . import anthropic_native, commonstack, openrouter

# Optional Anthropic SDK — kept here so provider modules stay free of the
# optional-import side effects / print noise from the harness.
try:
    from anthropic import Anthropic as _Anthropic

    HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover - exercised when SDK missing
    _Anthropic = None
    HAS_ANTHROPIC = False

PROVIDERS = {
    commonstack.INTEGRATION_ID: commonstack,
    openrouter.INTEGRATION_ID: openrouter,
    anthropic_native.INTEGRATION_ID: anthropic_native,
}

KNOWN_INTEGRATIONS = tuple(PROVIDERS.keys())


class LLMProviderConfigurationError(RuntimeError):
    """Raised when an explicitly requested provider client is unavailable."""


def resolve_integration(integration: Optional[str] = None) -> str:
    """Normalize an integration id, or auto-detect from env when omitted.

    Explicit values must be one of ``KNOWN_INTEGRATIONS``. When ``None``/empty,
    prefer CommonStack if ``COMMONSTACK_API_KEY`` is set, otherwise native
    Anthropic — matching the pre-OpenRouter gateway preference. OpenRouter is
    opt-in only (via config / kwarg).
    """
    if integration is not None and str(integration).strip():
        key = str(integration).strip().lower()
        if key not in PROVIDERS:
            raise ValueError(
                f"Unknown LLM integration {integration!r}. "
                f"Expected one of: {', '.join(KNOWN_INTEGRATIONS)}"
            )
        return key
    if os.getenv("COMMONSTACK_API_KEY"):
        return commonstack.INTEGRATION_ID
    return anthropic_native.INTEGRATION_ID


def make_llm_client(
    integration: Optional[str] = None,
    *,
    reasoning_effort: Optional[str] = None,
) -> Optional[Any]:
    """Create an Anthropic-compatible client for the chosen integration.

    Returns ``None`` when the SDK is missing or the integration's API key is
    unset / client init fails, so callers fall back to rule-based trading.
    """
    if not HAS_ANTHROPIC or _Anthropic is None:
        return None
    resolved = resolve_integration(integration)
    if resolved == openrouter.INTEGRATION_ID:
        return openrouter.make_client(
            _Anthropic,
            reasoning_effort=reasoning_effort,
        )
    return PROVIDERS[resolved].make_client(_Anthropic)


def default_model_name(integration: Optional[str] = None) -> str:
    """Default model slug for the resolved integration."""
    resolved = resolve_integration(integration)
    return PROVIDERS[resolved].default_model_name()


def ensure_llm_client_available(integration: Optional[str] = None) -> Any:
    """Construct a client without making a network request, or fail safely."""
    if not HAS_ANTHROPIC or _Anthropic is None:
        raise LLMProviderConfigurationError(
            "LLM provider client is unavailable because the Anthropic SDK is missing"
        )
    resolved = resolve_integration(integration)
    client = make_llm_client(resolved)
    if client is None:
        raise LLMProviderConfigurationError(
            f"LLM provider client is unavailable for integration {resolved!r}"
        )
    return client


__all__ = [
    "HAS_ANTHROPIC",
    "KNOWN_INTEGRATIONS",
    "LLMProviderConfigurationError",
    "PROVIDERS",
    "anthropic_native",
    "commonstack",
    "default_model_name",
    "ensure_llm_client_available",
    "make_llm_client",
    "openrouter",
    "resolve_integration",
]
