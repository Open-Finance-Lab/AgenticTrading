"""Hosted-agent runtime selection and dispatch.

``pipeline`` deliberately delegates to a caller-provided function so the
existing backtest decision path remains the source of truth. Other runtimes
implement the same small ``decide(context) -> {"actions": [...]}`` boundary and
still hand their actions to ATL's portfolio manager for execution.
"""

from __future__ import annotations

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


class UnsupportedAgentRuntime(ValueError):
    """Raised when an agent names a runtime ATL does not host."""


def normalize_runtime_type(value: Any) -> str:
    runtime_type = str(value or DEFAULT_RUNTIME_TYPE).strip().lower()
    if runtime_type not in SUPPORTED_RUNTIME_TYPES:
        raise UnsupportedAgentRuntime(f"Unsupported agent runtime: {runtime_type}")
    return runtime_type


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
        return dict(config)

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
    return normalized


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

    def decide(self, context: AgentRuntimeContext) -> Dict[str, List[Dict[str, Any]]]: ...


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
        if self.runtime_type == AI_HEDGE_FUND_RUNTIME_TYPE and self._runtime is None:
            from dashboard.backend.infrastructure.ai_hedge_fund.adapter import (
                AiHedgeFundRuntime,
            )

            self._runtime = AiHedgeFundRuntime(self.runtime_config)

    @property
    def calls(self) -> int:
        return int(getattr(self._runtime, "calls", 0) or 0)

    @property
    def model_name(self) -> Optional[str]:
        value = getattr(self._runtime, "model_name", None)
        return str(value) if value else None

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
