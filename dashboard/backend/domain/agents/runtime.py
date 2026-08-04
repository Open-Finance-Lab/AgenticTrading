"""Hosted-agent runtime selection and dispatch.

``pipeline`` deliberately delegates to a caller-provided function so the
existing backtest decision path remains the source of truth. Other runtimes
implement the same small ``decide(context) -> {"actions": [...]}`` boundary and
still hand their actions to ATL's portfolio manager for execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol

PIPELINE_RUNTIME_TYPE = "pipeline"
AI_HEDGE_FUND_RUNTIME_TYPE = "ai_hedge_fund"
SUPPORTED_RUNTIME_TYPES = frozenset(
    {PIPELINE_RUNTIME_TYPE, AI_HEDGE_FUND_RUNTIME_TYPE}
)
DEFAULT_RUNTIME_TYPE = PIPELINE_RUNTIME_TYPE

# Upstream's analyst registry at the pinned revision. Analyst composition is
# the one strategy-level choice persisted on an ATL agent; model/provider,
# interpreter, time-window, and timeout settings remain platform-owned.
AI_HEDGE_FUND_ANALYSTS = (
    "aswath_damodaran",
    "ben_graham",
    "bill_ackman",
    "cathie_wood",
    "charlie_munger",
    "michael_burry",
    "mohnish_pabrai",
    "nassim_taleb",
    "peter_lynch",
    "phil_fisher",
    "rakesh_jhunjhunwala",
    "stanley_druckenmiller",
    "warren_buffett",
    "technical_analyst",
    "fundamentals_analyst",
    "growth_analyst",
    "news_sentiment_analyst",
    "sentiment_analyst",
    "valuation_analyst",
)
_AI_HEDGE_FUND_ANALYST_SET = frozenset(AI_HEDGE_FUND_ANALYSTS)

# ``pipeline`` has no defined knobs yet, so its config stays permissive about
# *which* keys appear -- but permissive must not mean unbounded. Without a size
# ceiling any authenticated caller can PATCH megabytes of arbitrary JSON onto
# every agent they own, since the column is plain TEXT. The sibling ``pipeline``
# field is capped at 50 steps for the same reason; this is that cap's analogue.
# Generous next to real payloads: all 19 analysts serialize to ~400 bytes.
MAX_RUNTIME_CONFIG_BYTES = 8192


class UnsupportedAgentRuntime(ValueError):
    """Raised when an agent names a runtime ATL does not host."""


class AgentRuntimeError(RuntimeError):
    """A hosted runtime failed to produce a decision for one step.

    Runtime-agnostic on purpose: the dispatcher is the boundary the backtest
    engine handles failures at, so the engine never has to import (or know
    about) any particular runtime's adapter to keep a run alive.
    """


class AgentRuntimeConfigurationError(AgentRuntimeError):
    """The deployment cannot host this runtime at all.

    Separated from the transient base because it fails identically on every
    step. Callers should abort rather than spend the run's failure budget
    rediscovering the same missing interpreter once per trading day.
    """


def normalize_runtime_type(value: Any) -> str:
    runtime_type = str(value or DEFAULT_RUNTIME_TYPE).strip().lower()
    if runtime_type not in SUPPORTED_RUNTIME_TYPES:
        raise UnsupportedAgentRuntime(f"Unsupported agent runtime: {runtime_type}")
    return runtime_type


def _bounded_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reject configs that cannot be persisted safely as JSON text."""
    try:
        encoded = json.dumps(config, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_config must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_RUNTIME_CONFIG_BYTES:
        raise ValueError(
            f"runtime_config must be at most {MAX_RUNTIME_CONFIG_BYTES} bytes of JSON"
        )
    return config


def normalize_runtime_config(runtime_type: str, config: Any) -> Dict[str, Any]:
    """Validate the persisted, non-secret runtime knobs.

    Interpreter paths and credentials intentionally are not agent config. They
    are deployment-owned environment settings, preventing a stored agent from
    selecting arbitrary executables or smuggling secrets into run metadata.
    """
    runtime_type = normalize_runtime_type(runtime_type)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("runtime_config must be a JSON object")
    if runtime_type == PIPELINE_RUNTIME_TYPE:
        return _bounded_runtime_config(dict(config))

    allowed = {"analysts"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unsupported AI Hedge Fund runtime_config fields: {unknown}")

    normalized: Dict[str, Any] = {}
    if "analysts" in config:
        analysts = config["analysts"]
        if not isinstance(analysts, list) or not all(
            isinstance(item, str) and item.strip() for item in analysts
        ):
            raise ValueError("runtime_config.analysts must be an array of names")
        names = [item.strip() for item in analysts]
        if not names:
            raise ValueError("runtime_config.analysts must include at least one analyst")
        if len(names) != len(set(names)):
            raise ValueError("runtime_config.analysts must not contain duplicates")
        unsupported = sorted(set(names) - _AI_HEDGE_FUND_ANALYST_SET)
        if unsupported:
            raise ValueError(
                f"runtime_config.analysts contains unsupported analysts: {unsupported}"
            )
        normalized["analysts"] = names
    return _bounded_runtime_config(normalized)


@dataclass(frozen=True)
class AgentRuntimeContext:
    """The ATL state made available to a hosted decision runtime."""

    timestamp: datetime
    backtest_start_date: str
    symbols: List[str]
    cash: float
    total_equity: float
    positions: Mapping[str, int]
    entry_prices: Mapping[str, float]
    current_prices: Mapping[str, float]
    latest_market_date_before_decision: Optional[date] = None
    market: Mapping[str, Any] = field(default_factory=dict)


class AgentRuntime(Protocol):
    calls: int

    def decide(self, context: AgentRuntimeContext) -> Dict[str, List[Dict[str, Any]]]:
        """Return one step's ATL action envelope for ``context``."""


class RuntimeDispatcher:
    """Route one decision step to the persisted agent runtime."""

    def __init__(
        self,
        runtime_type: str = DEFAULT_RUNTIME_TYPE,
        runtime_config: Optional[Dict[str, Any]] = None,
        *,
        runtime: Optional[AgentRuntime] = None,
    ):
        self.runtime_type = normalize_runtime_type(runtime_type)
        self.runtime_config = normalize_runtime_config(
            self.runtime_type, runtime_config or {}
        )
        self._runtime = runtime
        if self.runtime_type != PIPELINE_RUNTIME_TYPE and self._runtime is None:
            # Hosted runtimes are supplied by the caller rather than imported
            # here. This module is domain code; importing a concrete adapter to
            # self-construct made domain and infrastructure import each other,
            # and doing it lazily inside __init__ only deferred that cycle
            # instead of removing it.
            raise UnsupportedAgentRuntime(
                f"A {self.runtime_type} dispatcher requires its runtime instance"
            )

    @property
    def calls(self) -> int:
        return int(getattr(self._runtime, "calls", 0) or 0)

    @property
    def model_name(self) -> Optional[str]:
        value = getattr(self._runtime, "model_name", None)
        return str(value) if value else None

    @property
    def decision_audit_rows(self) -> List[Dict[str, Any]]:
        rows = getattr(self._runtime, "decision_audit_rows", [])
        return list(rows or [])

    def record_latest_execution(self, executed_count: int) -> None:
        recorder = getattr(self._runtime, "record_latest_execution", None)
        if callable(recorder):
            recorder(executed_count)

    def dispatch(
        self,
        context: AgentRuntimeContext,
        *,
        pipeline_handler: Callable[[], Dict[str, List[Dict[str, Any]]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        if self.runtime_type == PIPELINE_RUNTIME_TYPE:
            return pipeline_handler()
        if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE and self._runtime is not None:
            return self._runtime.decide(context)
        raise UnsupportedAgentRuntime(f"Unsupported agent runtime: {self.runtime_type}")
