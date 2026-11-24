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
TRADINGAGENTS_RUNTIME_TYPE = "tradingagents"
SUPPORTED_RUNTIME_TYPES = frozenset(
    {
        PIPELINE_RUNTIME_TYPE,
        AI_HEDGE_FUND_RUNTIME_TYPE,
        TRADINGAGENTS_RUNTIME_TYPE,
    }
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

# TradingAgents' analyst registry at the pinned 0.3.x revision
# (``tradingagents/graph/setup.py``). Deliberately validated against its own
# roster rather than a pooled one: the two products share no analyst names, and
# an AI Hedge Fund name reaching ``setup_graph`` raises mid-run, after the agent
# has already been created and a backtest accepted.
TRADINGAGENTS_ANALYSTS = ("market", "social", "news", "fundamentals")
_TRADINGAGENTS_ANALYST_SET = frozenset(TRADINGAGENTS_ANALYSTS)

# ---------------------------------------------------------------------------
# Look-ahead safety of each TradingAgents analyst.
#
# A historical backtest is only meaningful if every input an analyst reads was
# available on the decision date. That is a property of the *vendor
# implementation*, not of the tool signature: upstream has tools that accept a
# ``curr_date`` and ignore it, and tools that take no date at all. So this
# table was built by reading, at the pinned revision
# ``a33fd4c0f134485a43553a2c23a63cb14adbd88f``, every tool each analyst binds
# and the vendor function behind it. Audited 2026-10-27.
#
# ``market`` is the only analyst whose entire surface is bounded by the
# requested date:
#   - get_stock_data -> y_finance.get_YFin_data_online: ticker.history(start,
#     end=end_date + 1d), so the frame stops at the requested day.
#   - get_indicators -> stockstats_utils.load_ohlcv: data[data.Date <=
#     curr_date] before any indicator is computed.
#   - get_verified_market_snapshot -> market_data_validator._verified_rows,
#     same curr_date bound.
#
# Everything in TRADINGAGENTS_LOOKAHEAD_EVIDENCE reads wall-clock-current data
# regardless of the date the run asks for. There is no config hook that removes
# a tool from an analyst (each analyst hard-codes its own ``tools`` list), and
# the social analyst does not even go through tool-calling -- it pre-fetches
# into the prompt before the model is invoked, so the model cannot decline it.
# Hence the only fail-closed control available at this layer is refusing the
# analyst, which is what ``tradingagents_lookahead_unsafe_analysts`` drives.
#
# Each entry is (contaminating tool, why it cannot be bounded).
TRADINGAGENTS_LOOKAHEAD_EVIDENCE: Dict[str, tuple] = {
    "social": (
        (
            "fetch_stocktwits_messages",
            "takes (ticker, limit, timeout) and no date; returns the newest "
            "cashtag messages as of the wall clock",
        ),
        (
            "fetch_reddit_posts",
            "takes (ticker, subreddits, limit_per_sub, timeout, "
            "inter_request_delay) and no date; labels its own output "
            "'in the past 7 days' relative to now",
        ),
    ),
    "news": (
        (
            "get_prediction_markets",
            "takes (topic, limit) and no date; polymarket._is_forward_looking "
            "filters against datetime.now(timezone.utc), so it returns today's "
            "implied probabilities for events that are still open today",
        ),
    ),
    "fundamentals": (
        (
            "get_fundamentals",
            "y_finance.get_fundamentals annotates curr_date as 'not used for "
            "yfinance' and returns live yf.Ticker().info -- current market cap, "
            "trailing/forward PE, and 52-week high/low plus 50- and 200-day "
            "averages, which encode prices after the decision date",
        ),
        (
            "get_balance_sheet / get_cashflow / get_income_statement",
            "curr_date defaults to None and filter_financials_by_date returns "
            "the frame unfiltered when it is absent, so a model that omits the "
            "argument sees future fiscal periods; when it is supplied the "
            "filter bounds the fiscal period END, not the filing date, so a "
            "quarter is visible weeks before it was published",
        ),
    ),
}

# The analysts a historical run may use. Derived from the audit above rather
# than written out twice, so adding an evidence entry removes the analyst.
TRADINGAGENTS_AS_OF_SAFE_ANALYSTS = tuple(
    name for name in TRADINGAGENTS_ANALYSTS
    if name not in TRADINGAGENTS_LOOKAHEAD_EVIDENCE
)


def tradingagents_lookahead_unsafe_analysts(analysts: Any) -> List[str]:
    """Return the selected analysts that cannot be bounded to a past date.

    Order follows ``TRADINGAGENTS_ANALYSTS`` so the message is stable
    regardless of how the caller ordered its roster.
    """
    selected = {
        str(name).strip()
        for name in (analysts or [])
        if isinstance(name, str) and str(name).strip()
    }
    return [
        name
        for name in TRADINGAGENTS_ANALYSTS
        if name in selected and name in TRADINGAGENTS_LOOKAHEAD_EVIDENCE
    ]


def describe_tradingagents_lookahead(analysts: Any) -> str:
    """Return a caller-facing explanation of why a roster is refused.

    Names the offending tool for each analyst: "unsafe analyst" alone gives the
    user nothing to act on, and the whole point of refusing instead of silently
    dropping the analyst is that the user learns what was wrong with their
    configuration.
    """
    unsafe = tradingagents_lookahead_unsafe_analysts(analysts)
    if not unsafe:
        return ""
    reasons = "; ".join(
        f"{name} ({', '.join(tool for tool, _ in TRADINGAGENTS_LOOKAHEAD_EVIDENCE[name])})"
        for name in unsafe
    )
    safe = ", ".join(TRADINGAGENTS_AS_OF_SAFE_ANALYSTS)
    return (
        "TradingAgents analysts that read data with no historical cutoff "
        f"cannot be used in a backtest: {reasons}. Their data sources return "
        "information as of today regardless of the date being analyzed, so "
        f"decisions built on them describe the future. Available: {safe}."
    )

# Analyst composition is the one strategy-level choice a hosted agent persists;
# every other knob (model, provider, interpreter, position sizing, timeouts)
# stays deployment-owned. Keyed by runtime so adding a hosted runtime cannot
# accidentally inherit another product's roster.
_HOSTED_RUNTIME_ANALYSTS = {
    AI_HEDGE_FUND_RUNTIME_TYPE: (_AI_HEDGE_FUND_ANALYST_SET, "AI Hedge Fund"),
    TRADINGAGENTS_RUNTIME_TYPE: (_TRADINGAGENTS_ANALYST_SET, "TradingAgents"),
}

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

    if runtime_type not in _HOSTED_RUNTIME_ANALYSTS:
        # Only reachable by adding a runtime to SUPPORTED_RUNTIME_TYPES and
        # forgetting its roster. A bare subscript raised KeyError here, which
        # every router maps to a 500 -- so the deployment's own omission was
        # reported to the user as a server fault with no indication of what
        # was wrong. test_every_hosted_runtime_declares_a_roster catches the
        # omission at test time; this is the runtime half.
        raise UnsupportedAgentRuntime(
            f"Runtime {runtime_type!r} declares no analyst roster; add one to "
            "_HOSTED_RUNTIME_ANALYSTS"
        )
    analyst_set, runtime_label = _HOSTED_RUNTIME_ANALYSTS[runtime_type]
    allowed = {"analysts"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported {runtime_label} runtime_config fields: {unknown}"
        )

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
        unsupported = sorted(set(names) - analyst_set)
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
    def decisions(self) -> int:
        """Steps the runtime actually drove a decision for.

        Distinct from ``calls``, which is the billing counter: an upstream
        response that fails validation is paid for but decided nothing.

        ⚠ A runtime that does not separate the two reports its call count,
        which is the previous behaviour -- and ``AiHedgeFundRuntime`` is such
        a runtime, so the honest-coverage property does **not** currently
        apply to it even though its ``decide`` has the same
        increment-then-map shape. The fallback is deliberate (removing it
        would change a figure the leaderboard integrity guard reads, which is
        a product decision) but it is a known gap, not a no-op.
        """
        value = getattr(self._runtime, "decisions", None)
        if value is None:
            return self.calls
        return int(value or 0)

    @property
    def sizing_filter_counts(self) -> Dict[str, int]:
        """Per-reason counts of decisions that produced no order."""
        counts = getattr(self._runtime, "sizing_filter_counts", None)
        return dict(counts or {})

    @property
    def skipped_steps(self) -> List[Dict[str, str]]:
        """Steps that produced no decision at all, and why."""
        return list(getattr(self._runtime, "skipped_steps", []) or [])

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
        if self._runtime is not None:
            # Every non-pipeline runtime reaches this line: __init__ has already
            # rejected an unsupported type and refused to build a hosted
            # dispatcher without its runtime instance. Branching per runtime
            # here would only add a place to forget one.
            return self._runtime.decide(context)
        raise UnsupportedAgentRuntime(f"Unsupported agent runtime: {self.runtime_type}")
