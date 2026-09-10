"""OpenRouter gateway integration.

OpenRouter exposes multi-provider models (including NVIDIA Nemotron) behind one
key. Its Anthropic Messages skin (``https://openrouter.ai/api``) returns the
same Anthropic response shape as CommonStack, so the shared backtest harness
works unchanged — we only swap ``base_url``, auth, and the ``provider/model``
slug.

Reasoning models (Nemotron Nano, etc.) emit ``thinking`` / ``redacted_thinking``
blocks by default. CommonStack leaderboard models (Gemini, Qwen, DeepSeek, …)
likewise bill thinking into ``output_tokens``, so OpenRouter enables reasoning
by default. Effort levels map to a hard ``reasoning.max_tokens`` budget (e.g.
medium→2048) so a JSON text block still fits on long trading prompts — bare
provider default / effort-% often spends the whole ceiling on thinking. That
budget is then clamped against each request's own ``max_tokens`` (see
``_clamp_reasoning_budget``): a mapped budget can exceed the configured output
ceiling — medium's 2048 against the harness's default 2000 does — and a request
whose thinking budget does not leave room for an answer cannot return one. Set
``OPENROUTER_REASONING_EFFORT=none`` to force-disable; ``auto`` leaves the
provider alone; ``OPENROUTER_REASONING_MAX_TOKENS`` overrides the mapped budget.
"""

from __future__ import annotations

import os
from typing import Any, Optional

INTEGRATION_ID = "openrouter"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_BASE_URL = "https://openrouter.ai/api"

# Match CommonStack thinking models: reasoning enabled by default.
# Effort levels map to a hard ``reasoning.max_tokens`` budget (OpenRouter docs:
# prefer max_tokens *or* effort, not both). A fixed budget is more reliable than
# effort-% of the request ceiling — on long trading prompts Nemotron otherwise
# often returns only thinking/redacted_thinking and no JSON text.
# Override with none|auto, or set OPENROUTER_REASONING_MAX_TOKENS explicitly.
_DEFAULT_REASONING_EFFORT = "medium"
_OFF_VALUES = frozenset({"none", "off", "false", "0", "disabled"})
_PASSTHROUGH_VALUES = frozenset({"auto", "default"})
# OpenRouter minimum for reasoning.max_tokens is 1024.
_EFFORT_TO_REASONING_BUDGET = {
    "minimal": 1024,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 6144,
    "max": 8192,
}

# A thinking budget and the answer share one output ceiling. Anthropic requires
# ``max_tokens > thinking.budget_tokens``, and a reply that spends the whole
# ceiling on reasoning carries no text block at all -- which is exactly the
# empty response the retry paths in ``pipeline_runner`` / ``portfolio_manager``
# exist to recover from, at one billed call each. The default pairing was
# structurally incapable of returning text: effort ``medium`` asks for 2048
# thinking tokens while ``request_trading_decision`` sends ``max_tokens=2000``.
# So the budget is clamped against the ceiling the caller is actually sending,
# leaving this much room for the JSON.
_MIN_ANSWER_TOKENS = 512
# OpenRouter's floor for ``reasoning.max_tokens``. A budget below it is not a
# smaller budget, it is a rejected request -- so a ceiling too small to hold
# the floor plus an answer disables reasoning instead of shrinking it.
_MIN_REASONING_BUDGET = 1024

# One line per distinct (budget, ceiling) pairing: this runs per request, and a
# silent clamp is the failure mode being fixed, not an acceptable one.
_WARNED_BUDGET_CEILINGS: set = set()


def _clamp_reasoning_budget(budget: int, max_tokens: Any) -> Optional[int]:
    """Largest legal thinking budget that still leaves room for an answer.

    ``None`` when the ceiling cannot fit ``_MIN_REASONING_BUDGET`` plus an
    answer: the caller then disables reasoning rather than sending a request
    that cannot return text. An unknown ceiling (``None``, or anything not a
    positive int) is left alone -- there is nothing to clamp against.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        return budget
    if max_tokens <= 0:
        return budget
    room = max_tokens - _MIN_ANSWER_TOKENS
    if budget <= room:
        return budget
    clamped = room if room >= _MIN_REASONING_BUDGET else None
    key = (budget, max_tokens)
    if key not in _WARNED_BUDGET_CEILINGS:
        _WARNED_BUDGET_CEILINGS.add(key)
        action = (
            f"clamping it to {clamped}"
            if clamped is not None
            else "disabling reasoning for this request"
        )
        print(
            f"⚠️ OpenRouter reasoning budget {budget} leaves no room for an "
            f"answer under max_tokens={max_tokens}; {action} so the reply can "
            f"still carry a text block"
        )
    return clamped


def _reasoning_off_body() -> dict[str, Any]:
    """``extra_body`` payload that disables reasoning (a fresh dict per call)."""
    # ``effort: none`` is the documented disable; ``enabled: false`` and
    # ``exclude: true`` cover providers that ignore effort alone.
    return {
        "reasoning": {
            "effort": "none",
            "enabled": False,
            "exclude": True,
        }
    }


def _reasoning_body_for_budget(budget: int, max_tokens: Any) -> dict[str, Any]:
    clamped = _clamp_reasoning_budget(budget, max_tokens)
    if clamped is None:
        return _reasoning_off_body()
    return {"reasoning": {"max_tokens": clamped, "enabled": True}}


def base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)


def default_model_name() -> str:
    return DEFAULT_MODEL


def _reasoning_effort(override: Optional[str] = None) -> str:
    raw = override
    if raw is None or not str(raw).strip():
        raw = os.getenv("OPENROUTER_REASONING_EFFORT", _DEFAULT_REASONING_EFFORT)
    return (raw or "").strip().lower()


def _reasoning_max_tokens_override() -> Optional[int]:
    raw = os.getenv("OPENROUTER_REASONING_MAX_TOKENS")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1024 else None


def reasoning_is_disabled(override: Optional[str] = None) -> bool:
    """True when we force non-reasoning (``OPENROUTER_REASONING_EFFORT=none``)."""
    effort = _reasoning_effort(override)
    return effort in _OFF_VALUES


def reasoning_is_explicitly_enabled(override: Optional[str]) -> bool:
    """True when a *config-supplied* effort turns extended thinking on.

    Deliberately environment-independent: only an explicit, non-passthrough
    effort counts. ``None`` means "let the env / provider decide", which is not
    something config validation can resolve, so it reads as "unknown", not
    "enabled".
    """
    if override is None or not str(override).strip():
        return False
    effort = str(override).strip().lower()
    return effort not in _OFF_VALUES and effort not in _PASSTHROUGH_VALUES


def reasoning_extra_body(
    override: Optional[str] = None,
    *,
    max_tokens: Any = None,
) -> Optional[dict[str, Any]]:
    """OpenRouter ``reasoning`` payload to merge into ``extra_body``, or ``None``.

    ``OPENROUTER_REASONING_EFFORT``:
      * unset → ``medium`` → ``reasoning.max_tokens=2048`` (reasoning on)
      * ``default`` / ``auto`` → no injection (raw provider default)
      * ``none`` / ``off`` → disable reasoning
      * ``low`` / ``medium`` / ``high`` / … → mapped max_tokens budget
      * unknown effort string → passed through as ``effort``

    ``OPENROUTER_REASONING_MAX_TOKENS`` (optional, ≥1024) overrides the mapped
    budget whenever reasoning is enabled.
    """
    effort = _reasoning_effort(override)
    if effort in _PASSTHROUGH_VALUES:
        return None
    if effort in _OFF_VALUES:
        return _reasoning_off_body()
    budget_override = _reasoning_max_tokens_override()
    if budget_override is not None:
        return _reasoning_body_for_budget(budget_override, max_tokens)
    budget = _EFFORT_TO_REASONING_BUDGET.get(effort)
    if budget is not None:
        return _reasoning_body_for_budget(budget, max_tokens)
    # An unrecognised effort string carries no number to clamp; the provider
    # owns the budget in that case.
    return {"reasoning": {"effort": effort, "enabled": True}}


def _reasoning_budget_tokens(override: Optional[str] = None) -> Optional[int]:
    """Resolved thinking budget (≥1024), or None when passthrough/disabled."""
    effort = _reasoning_effort(override)
    if effort in _OFF_VALUES:
        return None
    if effort in _PASSTHROUGH_VALUES:
        return None
    max_tokens_override = _reasoning_max_tokens_override()
    if max_tokens_override is not None:
        return max_tokens_override
    return _EFFORT_TO_REASONING_BUDGET.get(effort, 2048)


def anthropic_thinking_kwarg(
    override: Optional[str] = None,
    *,
    max_tokens: Any = None,
) -> Optional[dict[str, Any]]:
    """Anthropic Messages ``thinking`` kwarg for OpenRouter's Anthropic skin.

    When reasoning is disabled → ``{"type": "disabled"}``.
    When enabled with a known budget → ``{"type": "enabled", "budget_tokens": N}``
    so thinking cannot consume the entire ``max_tokens`` ceiling (Nemotron
    otherwise often returns only thinking/redacted_thinking).

    ``max_tokens`` is that ceiling. Pass it: a budget larger than the ceiling
    made the "so thinking cannot consume the entire ceiling" guarantee above
    false, and a budget it cannot share with an answer disables reasoning
    outright rather than shipping a request that cannot reply.
    """
    if reasoning_is_disabled(override):
        return {"type": "disabled"}
    budget = _reasoning_budget_tokens(override)
    if budget is None:
        return None
    clamped = _clamp_reasoning_budget(budget, max_tokens)
    if clamped is None:
        return {"type": "disabled"}
    return {"type": "enabled", "budget_tokens": clamped}


def _default_headers() -> dict[str, str]:
    """Optional OpenRouter ranking / attribution headers."""
    headers: dict[str, str] = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


class _OpenRouterMessages:
    """Proxy that injects OpenRouter reasoning-off defaults when unset by caller."""

    def __init__(self, inner: Any, reasoning_effort: Optional[str] = None):
        self._inner = inner
        self._reasoning_effort = reasoning_effort

    def create(self, **kwargs: Any) -> Any:
        # Both payloads are clamped against the ceiling this very request
        # carries, not against the default one: the recovery calls raise
        # ``max_tokens`` precisely so reasoning and the JSON both fit, and a
        # demo run lowers it below anything reasoning can share.
        ceiling = kwargs.get("max_tokens")
        extras = reasoning_extra_body(self._reasoning_effort, max_tokens=ceiling)
        if extras:
            body = dict(kwargs.get("extra_body") or {})
            if "reasoning" not in body:
                body.update(extras)
                kwargs["extra_body"] = body
        thinking = anthropic_thinking_kwarg(
            self._reasoning_effort, max_tokens=ceiling
        )
        if thinking is not None and "thinking" not in kwargs:
            kwargs["thinking"] = thinking
        return self._inner.create(**kwargs)


class OpenRouterClient:
    """Anthropic client wrapper with OpenRouter-specific ``messages.create`` defaults."""

    def __init__(self, client: Any, reasoning_effort: Optional[str] = None):
        self._client = client
        self.messages = _OpenRouterMessages(
            client.messages,
            reasoning_effort=reasoning_effort,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def make_client(
    anthropic_cls: Any,
    *,
    reasoning_effort: Optional[str] = None,
) -> Optional[Any]:
    """Build an Anthropic-compatible OpenRouter client, or ``None``."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    kwargs: dict[str, Any] = {
        "api_key": key,
        "base_url": base_url(),
    }
    headers = _default_headers()
    if headers:
        kwargs["default_headers"] = headers
    try:
        client = OpenRouterClient(
            anthropic_cls(**kwargs),
            reasoning_effort=reasoning_effort,
        )
        if reasoning_is_disabled(reasoning_effort):
            print(
                "ℹ️  OpenRouter: reasoning disabled "
                "(OPENROUTER_REASONING_EFFORT=none)"
            )
        else:
            effort = _reasoning_effort(reasoning_effort) or _DEFAULT_REASONING_EFFORT
            print(
                f"ℹ️  OpenRouter: reasoning on "
                f"(OPENROUTER_REASONING_EFFORT={effort}; "
                f"parity with CommonStack thinking models)"
            )
        return client
    except Exception as exc:  # pragma: no cover - defensive
        print(f"⚠️  Failed to init OpenRouter client: {exc}")
        return None
